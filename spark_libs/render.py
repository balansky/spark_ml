

def select_sql(sql_info):
    where_clause = []
    sql = "select " + sql_info['unique_id'] + ","
    sql += ','.join(sql_info['text_fields'])
    if sql_info['other_fields']:
        sql += ','.join(sql_info['other_fields'])
    if sql_info['label_field']:
        sql += ',' + sql_info['label_field'] + ' as label'
    sql += " from " + sql_info['table']
    if sql_info['conditional_fields']['pos']:
        where_clause.extend([key + " = " + val for key, val in sql_info['conditional_fields']['pos'].items()])
    if sql_info['conditional_fields']['neg']:
        where_clause.extend([key + " != " + val for key, val in sql_info['conditional_fields']['neg'].items()])
    if where_clause:
        sql += " where " + " and ".join(where_clause)
    if sql_info['size']:
        sql += " limit " + str(sql_info['size'])
    return sql

def update_prod_sql(update_info):
    sql = "update " + update_info['table'] + " set "
    where_clause = []
    update_clause = [update_field + " = %s" for update_field in update_info['update_fields'].keys()]
    if update_info['conditional_fields']['pos']:
        where_clause.extend([ key + " = %s"  for key in update_info['conditional_fields']['pos'].keys()])
    if update_info['conditional_fields']['neg']:
        where_clause.extend([ key + " != %s"  for key in update_info['conditional_fields']['neg'].keys()])
    sql += ",".join(update_clause)
    if where_clause:
        sql += " where " + " and ".join(where_clause)
    sql += ";"
    return sql
