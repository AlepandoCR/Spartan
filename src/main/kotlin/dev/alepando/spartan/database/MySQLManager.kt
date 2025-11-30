package dev.alepando.spartan.database

import com.google.gson.Gson
import dev.alepando.spartan.Spartan
import java.sql.Connection
import java.sql.DriverManager
import java.sql.SQLException

class MySQLManager(val plugin: Spartan) {

    private val config = plugin.config
    private val host: String = config.getString("mysql.host")!!
    private val port: Int = config.getInt("mysql.port")
    private val database: String = config.getString("mysql.database")!!
    private val user: String = config.getString("mysql.user")!!
    private val password: String = config.getString("mysql.password")!!
    private val useSSL: Boolean = config.getBoolean("mysql.useSSL")

    private var connection: Connection? = null
    val gson = Gson()

    fun connect() {
        if (connection != null && !connection!!.isClosed) return

        val url = "jdbc:mysql://$host:$port/$database?useSSL=$useSSL&autoReconnect=true"
        try {
            connection = DriverManager.getConnection(url, user, password)
            plugin.logger.info("Connected to MySQL!")
        } catch (ex: SQLException) {
            ex.printStackTrace()
        }
    }

    fun disconnect() {
        try {
            connection?.close()
            plugin.logger.info("Disconnected from MySQL!")
        } catch (ex: SQLException) {
            ex.printStackTrace()
        }
    }

    fun getConnection(): Connection {
        if (connection == null || connection!!.isClosed) {
            connect()
        }
        return connection!!
    }
}