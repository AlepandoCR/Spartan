package dev.alepando.spartan.database.table.types

import dev.alepando.spartan.database.MySQLManager
import dev.alepando.spartan.database.data.DqnDto
import dev.alepando.spartan.database.table.AutoTable

/** Table wrapper for serialized DQN models keyed by [hash]. */
class DqnModelTable(mysql: MySQLManager)
    : AutoTable<DqnDto>(mysql, DqnDto::class, "dqn_models", primaryKey = "hash")
