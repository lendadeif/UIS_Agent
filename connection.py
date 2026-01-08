from sqlalchemy import create_engine
from urllib.parse import quote_plus
from langchain_community.utilities.sql_database import SQLDatabase

class DatabaseManager:
    _db_instance = None

    allowed_tables = [
        "tab_Bill",
        "tab_Bill_detail",
        "tab_CallCenterOrder",
        "tab_CallCenterOrderDetail",
        "tab_company",
        "tab_Complaints",
        "tab_CompanyCurrency",
        "Customer",
        "Driver",
        "tab_frenchise",
        "tab_item",
        "tab_Site"
    ]

    @staticmethod
    def get_db():
        server = "18.224.175.116"
        database = "DB_A45ECB_ResErpDBHighTest"
        username = "AI"
        password = "AI@123"

        if DatabaseManager._db_instance is None:
            params = quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                "Encrypt=yes;"
                "TrustServerCertificate=yes;"
            )

            uri = f"mssql+pyodbc:///?odbc_connect={params}"

            engine = create_engine(uri, pool_pre_ping=True)

            DatabaseManager._db_instance = SQLDatabase(
                engine,
                include_tables=DatabaseManager.allowed_tables,
                sample_rows_in_table_info=2,
                view_support=False
            )

        return DatabaseManager._db_instance
    
