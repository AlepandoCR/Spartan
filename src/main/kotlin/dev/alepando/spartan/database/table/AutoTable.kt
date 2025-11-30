package dev.alepando.spartan.database.table

import com.google.gson.reflect.TypeToken
import database.table.AbstractTable
import dev.alepando.spartan.database.MySQLManager
import java.util.*
import kotlin.reflect.KClass
import kotlin.reflect.KType
import kotlin.reflect.full.memberProperties
import java.lang.reflect.Type

abstract class AutoTable<T : Any>(
    mysql: MySQLManager,
    private val clazz: KClass<T>,
    private val tableName: String,
    private val primaryKey: String = "uuid"
) : AbstractTable(mysql) {

    private val columns: List<Column<*>> = clazz.memberProperties.map {
        val isNullable = it.returnType.isMarkedNullable
        val isPrimaryKey = it.name == primaryKey
        val javaType = (it.returnType.classifier as? KClass<*>)?.java
            ?: error("Cannot resolve Java type for property: ${it.name}")

        Column(
            name = it.name,
            type = javaType,
            primaryKey = isPrimaryKey,
            nullable = isNullable
        )
    }

    override fun tableName() = tableName

    override fun createTable() {
        val sql = buildString {
            append("CREATE TABLE IF NOT EXISTS `$tableName` (\n")
            append(columns.joinToString(",\n") { it.definition() })
            append("\n);")
        }

        connection.createStatement().use {
            it.execute(sql)
        }
    }

    fun insertOrUpdate(obj: T) {
        val gson = mysql.gson.newBuilder().serializeSpecialFloatingPointValues().create()
        val props = clazz.memberProperties
        val names = props.joinToString(",") { "`" + it.name + "`" }
        val values = props.map { prop ->
            when (val value = prop.get(obj)) {
                null -> "NULL"
                is String, is UUID -> "'$value'"
                is Number, is Boolean -> "$value"
                else -> "'${gson.toJson(value)}'"
            }
        }

        val valuesStr = values.joinToString(",")
        val sql = "REPLACE INTO `$tableName` ($names) VALUES ($valuesStr);"

        connection.createStatement().use {
            it.executeUpdate(sql)
        }
    }

    fun findBy(field: String, value: Any): T? {
        val sql = "SELECT * FROM `$tableName` WHERE `$field` = ? LIMIT 1;"
        connection.prepareStatement(sql).use { statement ->
            statement.setString(1, value.toString())
            val rs = statement.executeQuery()

            return if (rs.next()) {
                val constructor = clazz.constructors.first()
                val args = constructor.parameters.map { param ->
                    val name = param.name ?: error("Missing parameter name for constructor")
                    val column = columns.find { it.name == name } ?: error("Column not found: $name")
                    val raw = rs.getObject(name)

                    val prop = clazz.memberProperties.find { it.name == param.name }!!
                    val type = prop.returnType
                    when {
                        raw == null -> null
                        column.type == UUID::class.java -> UUID.fromString(raw.toString())
                        column.type == String::class.java -> raw.toString()
                        column.type == Int::class.java || column.type == Integer::class.java -> (raw as Number).toInt()
                        column.type == Long::class.java -> (raw as Number).toLong()
                        column.type == Boolean::class.java -> raw as Boolean
                        column.type == List::class.java -> {
                            val listType = getType(type)
                            mysql.gson.fromJson(raw.toString(), listType)
                        }
                        else -> mysql.gson.fromJson(raw.toString(), column.type)
                    }
                }

                constructor.call(*args.toTypedArray())
            } else null
        }
    }

    private fun getType(type: KType): Type {
        val classifier = type.classifier as? KClass<*> ?: error("Unknown classifier: $type")
        val javaType = when (classifier) {
            Double::class -> java.lang.Double::class.java
            Float::class -> java.lang.Float::class.java
            Int::class -> Integer::class.java
            Long::class -> java.lang.Long::class.java
            Boolean::class -> java.lang.Boolean::class.java
            else -> classifier.java
        }

        if (type.arguments.isEmpty()) {
            return javaType
        }

        val innerType = getType(type.arguments.first().type!!)
        return TypeToken.getParameterized(javaType, innerType).type
    }

}