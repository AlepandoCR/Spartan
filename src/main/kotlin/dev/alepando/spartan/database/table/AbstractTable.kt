package database.table

import dev.alepando.spartan.database.MySQLManager
import java.sql.Connection

abstract class AbstractTable(protected val mysql: MySQLManager) {

    val connection: Connection get() = mysql.getConnection()

    abstract fun createTable()

    open fun dropTable() {
        val statement = connection.createStatement()
        statement.execute("DROP TABLE IF EXISTS ${tableName()}")
        statement.close()
    }

    abstract fun tableName(): String
}