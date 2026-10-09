import sqlite3

import sqlite_vec


def load_vec(connection: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension; False when this interpreter cannot."""
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
    except sqlite3.Error, AttributeError:
        return False
    return True
