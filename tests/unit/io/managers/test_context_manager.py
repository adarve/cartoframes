from collections import namedtuple

import pytest
import requests

from carto.datasets import DatasetManager
from carto.sql import SQLClient, BatchSQLClient, CopySQLClient
from carto.exceptions import CartoException, CartoRateLimitException

from pandas import DataFrame
from geopandas import GeoDataFrame
from cartoframes.auth import Credentials
from cartoframes.io.managers.context_manager import (
    ContextManager,
    DEFAULT_RETRY_TIMES,
    DEFAULT_STREAM_CHUNK_SIZE,
    BATCH_API_PAYLOAD_THRESHOLD,
    WIDE_COPY_COLUMN_THRESHOLD,
    _alter_table_drop_add_columns_query,
    _build_copy_from_query,
    _compute_copy_data,
    _explicit_copy_query_length,
    _is_wide_copy,
    _reorder_dataframe_columns,
    _stream_copy_data,
    retry_copy
)
from cartoframes.utils.columns import ColumnInfo


class TestContextManager(object):

    def setup_method(self):
        self.credentials = Credentials('fake_user', 'fake_api')

    def test_execute_query(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mock = mocker.patch.object(SQLClient, 'send')

        # When
        cm = ContextManager(self.credentials)
        cm.execute_query('query')

        # Then
        mock.assert_called_once_with('query', True, True, None)

    def test_execute_long_running_query(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mock = mocker.patch.object(BatchSQLClient, 'create_and_wait_for_completion')

        # When
        cm = ContextManager(self.credentials)
        cm.execute_long_running_query('query')

        # Then
        mock.assert_called_once_with('query')

    def test_copy_to(self, mocker):
        # Given
        query = '__query__'
        columns = [ColumnInfo('A', 'a', 'bigint', False)]
        mocker.patch.object(ContextManager, 'compute_query', return_value=query)
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=columns)
        mock = mocker.patch.object(ContextManager, '_copy_to')

        # When
        cm = ContextManager(self.credentials)
        cm.copy_to(query)

        # Then
        mock.assert_called_once_with('SELECT "A" FROM (__query__) _q', columns, 3)

    def test_copy_from(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=False)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        mock_create_table = mocker.patch.object(ContextManager, 'execute_query')
        mock = mocker.patch.object(ContextManager, '_copy_from')
        df = DataFrame({'A': [1]})
        columns = [ColumnInfo('A', 'a', 'bigint', False)]

        # When
        cm = ContextManager(self.credentials)
        cm.copy_from(df, 'TABLE NAME')

        # Then
        mock_create_table.assert_called_once_with('''
            BEGIN; CREATE TABLE table_name ("a" bigint); COMMIT;
        '''.strip())
        mock.assert_called_once()
        assert mock.call_args[0][1] == 'table_name'
        assert mock.call_args[0][2] == columns
        assert mock.call_args[0][3] == DEFAULT_RETRY_TIMES
        assert mock.call_args[1]['implicit_column_order'] is True

    def test_copy_from_exists_fail(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        df = DataFrame({'A': [1]})

        # When
        with pytest.raises(Exception) as e:
            cm = ContextManager(self.credentials)
            cm.copy_from(df, 'TABLE NAME', 'fail')

        # Then
        assert str(e.value) == ('Table "schema.table_name" already exists in your CARTO account. '
                                'Please choose a different `table_name` or use '
                                'if_exists="replace" to overwrite it.')

    def test_copy_from_exists_replace_truncate_and_drop_add_columns(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=[])
        mock = mocker.patch.object(ContextManager, '_truncate_and_drop_add_columns')
        mock_copy = mocker.patch.object(ContextManager, '_copy_from')
        df = DataFrame({'A': [1]})
        columns = [ColumnInfo('A', 'a', 'bigint', False)]

        # When
        cm = ContextManager(self.credentials)
        cm.copy_from(df, 'TABLE NAME', 'replace')

        # Then
        mock.assert_called_once_with('table_name', 'schema', columns, [])
        mock_copy.assert_called_once()
        assert mock_copy.call_args[0][1] == 'table_name'
        assert mock_copy.call_args[0][2] == columns
        assert mock_copy.call_args[0][3] == DEFAULT_RETRY_TIMES
        assert mock_copy.call_args[1]['implicit_column_order'] is False

    def test_copy_from_exists_replace_truncate(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        table_columns = [ColumnInfo('A', 'a', 'bigint', False)]
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=table_columns)
        mocker.patch.object(ContextManager, '_compare_columns', return_value=True)
        mock = mocker.patch.object(ContextManager, '_truncate_table')
        mock_copy = mocker.patch.object(ContextManager, '_copy_from')
        df = DataFrame({'A': [1]})
        columns = [ColumnInfo('A', 'a', 'bigint', False)]

        # When
        cm = ContextManager(self.credentials)
        cm.copy_from(df, 'TABLE NAME', 'replace')

        # Then
        mock.assert_called_once_with('table_name', 'schema')
        mock_copy.assert_called_once()
        assert mock_copy.call_args[0][1] == 'table_name'
        assert mock_copy.call_args[0][2] == columns
        assert mock_copy.call_args[0][3] == DEFAULT_RETRY_TIMES
        assert mock_copy.call_args[1]['implicit_column_order'] is False

    def test_internal_copy_from(self, mocker):
        # Given
        from shapely.geometry import Point
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mock = mocker.patch.object(CopySQLClient, 'copyfrom')
        gdf = GeoDataFrame({'A': [1, 2], 'B': [Point(0, 0), Point(1, 1)]})
        columns = [
            ColumnInfo('A', 'a', 'bigint', False),
            ColumnInfo('B', 'b', 'geometry', True)
        ]

        # When
        cm = ContextManager(self.credentials)
        cm._copy_from(gdf, 'table_name', columns)

        # Then
        assert mock.call_args[0][0] == '''
            COPY table_name("a","b") FROM stdin WITH (FORMAT csv, DELIMITER '|', NULL '__null');
        '''.strip()
        uploaded_data = b''.join(mock.call_args[0][1])
        assert uploaded_data == (
            b'1|0101000020E610000000000000000000000000000000000000\n'
            b'2|0101000020E6100000000000000000F03F000000000000F03F\n'
        )

    def test_rename_table(self, mocker):
        # Given
        def has_table(table_name):
            if table_name == 'table_name':
                return True
            elif table_name == 'new_table_name':
                return False
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', side_effect=has_table)
        mock = mocker.patch.object(ContextManager, '_rename_table')

        # When
        cm = ContextManager(self.credentials)
        result = cm.rename_table('table_name', 'NEW TABLE NAME')

        # Then
        mock.assert_called_once_with('table_name', 'new_table_name')
        assert result == 'new_table_name'

    def test_rename_table_equal(self, mocker):
        # When
        with pytest.raises(Exception) as e:
            cm = ContextManager(self.credentials)
            cm.rename_table('table_name', 'TABLE NAME')

        # Then
        assert str(e.value) == ('Table names are equal. Please choose a different table name.')

    def test_rename_table_orig_not_exist(self, mocker):
        # Given
        def has_table(table_name):
            if table_name == 'table_name':
                return False
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', side_effect=has_table)

        # When
        with pytest.raises(Exception) as e:
            cm = ContextManager(self.credentials)
            cm.rename_table('table_name', 'NEW TABLE NAME')

        # Then
        assert str(e.value) == ('Table "table_name" does not exist in your CARTO account.')

    def test_rename_table_dest_exists_fail(self, mocker):
        # Given
        def has_table(table_name):
            if table_name == 'table_name':
                return True
            elif table_name == 'new_table_name':
                return True
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', side_effect=has_table)

        # When
        with pytest.raises(Exception) as e:
            cm = ContextManager(self.credentials)
            cm.rename_table('table_name', 'NEW TABLE NAME', 'fail')

        # Then
        assert str(e.value) == ('Table "new_table_name" already exists in your CARTO account. '
                                'Please choose a different `new_table_name` or use '
                                'if_exists="replace" to overwrite it.')

    def test_rename_table_dest_exists_replace(self, mocker):
        # Given
        def has_table(table_name):
            if table_name == 'table_name':
                return True
            elif table_name == 'new_table_name':
                return True
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', side_effect=has_table)
        mock = mocker.patch.object(ContextManager, '_rename_table')

        # When
        cm = ContextManager(self.credentials)
        result = cm.rename_table('table_name', 'NEW TABLE NAME', 'replace')

        # Then
        mock.assert_called_once_with('table_name', 'new_table_name')
        assert result == 'new_table_name'

    def test_list_tables(self, mocker):
        # Given
        Dataset = namedtuple('Dataset', ['name', 'updated_at'])

        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(DatasetManager, 'filter', return_value=[
            Dataset('table_zero', 1), Dataset('table_one', 0)
        ])

        # When
        cm = ContextManager(self.credentials)
        tables = cm.list_tables()

        # Then
        assert DataFrame(['table_zero', 'table_one'], columns=['tables']).equals(tables)

    def test_list_tables_empty(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(DatasetManager, 'filter', return_value=[])

        # When
        cm = ContextManager(self.credentials)
        tables = cm.list_tables()

        # Then
        assert DataFrame(columns=['tables']).equals(tables)

    def test_retry_copy_decorator(self):
        @retry_copy
        def test_function(retry_times):
            class ResponseMock:
                def __init__(self):
                    self.text = 'My text'
                    self.headers = {
                        'Carto-Rate-Limit-Limit': 1,
                        'Carto-Rate-Limit-Remaining': 1,
                        'Retry-After': 1,
                        'Carto-Rate-Limit-Reset': 1
                    }
            response_mock = ResponseMock()
            raise CartoRateLimitException(response_mock)

        with pytest.raises(CartoRateLimitException):
            test_function(retry_times=1)

    def test_retry_copy_decorator_transient_error(self, mocker):
        mock_sleep = mocker.patch('cartoframes.io.managers.context_manager.time.sleep')
        attempts = {'count': 0}

        @retry_copy
        def test_function(retry_times):
            attempts['count'] += 1
            if attempts['count'] == 1:
                raise requests.exceptions.ChunkedEncodingError('connection broken')
            return 'ok'

        result = test_function(retry_times=2)

        assert result == 'ok'
        assert attempts['count'] == 2
        mock_sleep.assert_called_once()

    def test_build_copy_from_query_omits_columns_when_requested(self):
        columns = [ColumnInfo('A', 'a', 'bigint', False)]

        explicit_query = _build_copy_from_query('table_name', columns, use_explicit_columns=True)
        implicit_query = _build_copy_from_query('table_name', columns, use_explicit_columns=False)

        assert '("a")' in explicit_query
        assert '("a")' not in implicit_query
        assert implicit_query == (
            "COPY table_name FROM stdin WITH (FORMAT csv, DELIMITER '|', NULL '__null');")

    def test_build_copy_from_query_length_scales_with_columns(self):
        columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(1200)
        ]

        query = _build_copy_from_query('table_name', columns, use_explicit_columns=True)

        assert len(query) > BATCH_API_PAYLOAD_THRESHOLD

    def test_is_wide_copy_detects_column_count_threshold(self):
        columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(WIDE_COPY_COLUMN_THRESHOLD + 1)
        ]

        assert _explicit_copy_query_length('table_name', columns) < BATCH_API_PAYLOAD_THRESHOLD
        assert _is_wide_copy('table_name', columns) is True

    def test_internal_copy_from_uses_implicit_query_for_wide_table(self, mocker):
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mock = mocker.patch.object(CopySQLClient, 'copyfrom')
        columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(1200)
        ]
        df = DataFrame({column.name: ['value'] for column in columns})

        cm = ContextManager(self.credentials)
        cm._copy_from(df, 'table_name', columns, implicit_column_order=True)

        assert mock.call_args[0][0] == (
            "COPY table_name FROM stdin WITH (FORMAT csv, DELIMITER '|', NULL '__null');")

    def test_internal_copy_from_uses_implicit_query_for_many_columns(self, mocker):
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mock = mocker.patch.object(CopySQLClient, 'copyfrom')
        columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(WIDE_COPY_COLUMN_THRESHOLD + 1)
        ]
        df = DataFrame({column.name: ['value'] for column in columns})

        cm = ContextManager(self.credentials)
        cm._copy_from(df, 'table_name', columns, implicit_column_order=True)

        assert mock.call_args[0][0] == (
            "COPY table_name FROM stdin WITH (FORMAT csv, DELIMITER '|', NULL '__null');")

    def test_stream_copy_data_splits_wide_rows(self):
        columns = [ColumnInfo('A', 'a', 'text', False)]
        df = DataFrame({'A': ['x' * (DEFAULT_STREAM_CHUNK_SIZE * 2)]})

        row_data = b''.join(_compute_copy_data(df, columns))
        streamed_data = b''.join(_stream_copy_data(df, columns, chunk_size=DEFAULT_STREAM_CHUNK_SIZE))

        chunks = list(_stream_copy_data(df, columns, chunk_size=DEFAULT_STREAM_CHUNK_SIZE))
        assert len(chunks) > 1
        assert all(len(chunk) <= DEFAULT_STREAM_CHUNK_SIZE for chunk in chunks[:-1])
        assert streamed_data == row_data

    def test_copy_from_replace_recreates_wide_table(self, mocker):
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        table_columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(1200)
        ]
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=table_columns)
        mocker.patch.object(ContextManager, '_compare_columns', return_value=True)
        mock_recreate = mocker.patch.object(ContextManager, '_recreate_table_from_dataframe_columns')
        mock_truncate = mocker.patch.object(ContextManager, '_truncate_table')
        mock_copy = mocker.patch.object(ContextManager, '_copy_from')
        columns = table_columns
        df = DataFrame({column.name: ['value'] for column in columns})

        cm = ContextManager(self.credentials)
        cm.copy_from(df, 'TABLE NAME', 'replace')

        mock_recreate.assert_called_once_with('table_name', 'schema', columns)
        mock_truncate.assert_not_called()
        assert mock_copy.call_args[1]['implicit_column_order'] is True

    def test_copy_from_replace_recreates_many_column_table(self, mocker):
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        table_columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(WIDE_COPY_COLUMN_THRESHOLD + 1)
        ]
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=table_columns)
        mocker.patch.object(ContextManager, '_compare_columns', return_value=True)
        mock_recreate = mocker.patch.object(ContextManager, '_recreate_table_from_dataframe_columns')
        mock_truncate = mocker.patch.object(ContextManager, '_truncate_table')
        mock_copy = mocker.patch.object(ContextManager, '_copy_from')
        columns = table_columns
        df = DataFrame({column.name: ['value'] for column in columns})

        cm = ContextManager(self.credentials)
        cm.copy_from(df, 'TABLE NAME', 'replace')

        mock_recreate.assert_called_once_with('table_name', 'schema', columns)
        mock_truncate.assert_not_called()
        assert mock_copy.call_args[1]['implicit_column_order'] is True

    def test_copy_from_append_raises_for_wide_table(self, mocker):
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, 'has_table', return_value=True)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        columns = [
            ColumnInfo('col_{}'.format(i), 'col_{}'.format(i), 'text', False)
            for i in range(1200)
        ]
        mocker.patch.object(ContextManager, '_get_query_columns_info', return_value=columns)
        df = DataFrame({column.name: ['value'] for column in columns})

        with pytest.raises(CartoException) as error:
            cm = ContextManager(self.credentials)
            cm.copy_from(df, 'TABLE NAME', 'append')

        assert 'Cannot append a wide table' in str(error.value)

    def test_retry_copy_decorator_carto_exception_wrapped_error(self, mocker):
        mock_sleep = mocker.patch('cartoframes.io.managers.context_manager.time.sleep')
        attempts = {'count': 0}

        @retry_copy
        def test_function(retry_times):
            attempts['count'] += 1
            if attempts['count'] == 1:
                raise CartoException(requests.exceptions.ChunkedEncodingError('connection broken'))
            return 'ok'

        result = test_function(retry_times=2)

        assert result == 'ok'
        assert attempts['count'] == 2
        mock_sleep.assert_called_once()

    def test_retry_copy_decorator_read_timeout(self, mocker):
        mock_sleep = mocker.patch('cartoframes.io.managers.context_manager.time.sleep')
        attempts = {'count': 0}

        @retry_copy
        def test_function(retry_times):
            attempts['count'] += 1
            if attempts['count'] == 1:
                raise requests.exceptions.ReadTimeout('read timed out')
            return 'ok'

        result = test_function(retry_times=2)

        assert result == 'ok'
        assert attempts['count'] == 2
        mock_sleep.assert_called_once()

    def test_retry_copy_decorator_context_wrapped_error(self, mocker):
        mock_sleep = mocker.patch('cartoframes.io.managers.context_manager.time.sleep')
        attempts = {'count': 0}

        @retry_copy
        def test_function(retry_times):
            attempts['count'] += 1
            if attempts['count'] == 1:
                try:
                    raise requests.exceptions.ChunkedEncodingError('connection broken')
                except requests.exceptions.ChunkedEncodingError:
                    raise CartoException('copy failed')
            return 'ok'

        result = test_function(retry_times=2)

        assert result == 'ok'
        assert attempts['count'] == 2
        mock_sleep.assert_called_once()

    def test_explicit_copy_query_length_matches_build_query(self):
        columns = [ColumnInfo('A', 'a', 'bigint', False)]

        assert _explicit_copy_query_length('table_name', columns) == len(
            _build_copy_from_query('table_name', columns, use_explicit_columns=True))

    def test_reorder_dataframe_columns_drops_columns_not_in_target_schema(self):
        columns = [
            ColumnInfo('B', 'b', 'text', False),
            ColumnInfo('A', 'a', 'bigint', False)
        ]
        df = DataFrame({'A': [1], 'EXTRA': ['ignore'], 'B': ['value']})

        result = _reorder_dataframe_columns(df, columns)

        assert list(result.columns) == ['B', 'A']
        assert result.iloc[0].to_dict() == {'B': 'value', 'A': 1}

    def test_create_table_from_query_cartodbfy(self, mocker):
        # Given
        mocker.patch.object(ContextManager, 'has_table', return_value=False)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        mock = mocker.patch.object(ContextManager, 'execute_long_running_query')

        # When
        cm = ContextManager(self.credentials)
        cm.create_table_from_query('SELECT * FROM table_name', '__new_table_name__', if_exists='fail', cartodbfy=True)

        # Then
        mock.assert_called_with("SELECT CDB_CartodbfyTable('schema', '__new_table_name__')")

    def test_create_table_from_query_cartodbfy_default(self, mocker):
        # Given
        mocker.patch.object(ContextManager, 'has_table', return_value=False)
        mocker.patch.object(ContextManager, 'get_schema', return_value='schema')
        mock = mocker.patch.object(ContextManager, 'execute_long_running_query')

        # When
        cm = ContextManager(self.credentials)
        cm.create_table_from_query('SELECT * FROM table_name', '__new_table_name__', if_exists='fail')

        # Then
        mock.assert_called_with("SELECT CDB_CartodbfyTable('schema', '__new_table_name__')")

    def test_truncate_drop_add_columns_builds_query_without_double_semicolon(self, mocker):
        # Given
        mocker.patch('cartoframes.io.managers.context_manager._create_auth_client')
        mocker.patch.object(ContextManager, '_check_regenerate_table_exists', return_value=True)
        mock = mocker.patch.object(ContextManager, 'execute_long_running_query')
        df_columns = [
            ColumnInfo('SET', 'set', 'text', False),
            ColumnInfo('SET2', 'set2', 'text', False)
        ]
        table_columns = [
            ColumnInfo('SET', 'set', 'text', False)
        ]

        # When
        cm = ContextManager(self.credentials)
        cm._truncate_and_drop_add_columns('test_table', 'support', df_columns, table_columns)

        # Then
        query = mock.call_args[0][0]
        assert ';;' not in query
        assert query.startswith("SELECT CDB_RegenerateTable('support.test_table'::regclass); BEGIN;")


class TestAlterTableDropAddColumnsQuery(object):
    def test_builds_alter_table_with_drop_and_add(self):
        drop_columns = [
            ColumnInfo('SET', 'set', 'text', False)
        ]
        add_columns = [
            ColumnInfo('SET', 'set', 'text', False),
            ColumnInfo('SET2', 'set2', 'text', False)
        ]

        query = _alter_table_drop_add_columns_query(
            'test_table', drop_columns, add_columns)

        assert query == (
            'ALTER TABLE test_table DROP COLUMN "set",'
            'ADD COLUMN "set" text,ADD COLUMN "set2" text'
        )

    def test_builds_alter_table_with_only_add(self):
        add_columns = [
            ColumnInfo('SET2', 'set2', 'text', False)
        ]

        query = _alter_table_drop_add_columns_query(
            'test_table', [], add_columns)

        assert query == 'ALTER TABLE test_table ADD COLUMN "set2" text'

    def test_builds_alter_table_with_only_drop(self):
        drop_columns = [
            ColumnInfo('SET', 'set', 'text', False)
        ]

        query = _alter_table_drop_add_columns_query(
            'test_table', drop_columns, [])

        assert query == 'ALTER TABLE test_table DROP COLUMN "set"'

    def test_raises_if_no_operations(self):
        with pytest.raises(ValueError) as error:
            _alter_table_drop_add_columns_query('test_table', [], [])

        assert str(error.value) == 'No columns provided to drop or add.'
