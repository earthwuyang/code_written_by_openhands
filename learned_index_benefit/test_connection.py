import pymysql

# 数据库连接配置
host = "172.17.0.1"  # 远程服务器 IP 或本地 "localhost"
user = "user1"     # MySQL 用户名
password = "your_password" # MySQL 密码
database = "tpch_sf1" # 需要连接的数据库
port = 22224               # MySQL 默认端口

try:
    # 1. 连接到 MySQL 服务器
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        charset="utf8mb4",  # 设置字符编码
        cursorclass=pymysql.cursors.DictCursor,  # 以字典格式返回查询结果
    )

    print("✅ 成功连接到 MySQL 数据库！")

    # 2. 创建游标对象
    with connection.cursor() as cursor:
        # 执行 SQL 查询
        sql = "SELECT * FROM customer LIMIT 5;"  # 替换成你的表名
        cursor.execute(sql)

        # 获取查询结果
        results = cursor.fetchall()
        for row in results:
            print(row)  # 打印每一行数据

    # 3. 提交事务（如果执行的是 `INSERT`, `UPDATE`, `DELETE` 操作）
    connection.commit()

except pymysql.MySQLError as e:
    print("❌ 发生错误：", e)

finally:
    # 4. 关闭数据库连接
    if connection:
        connection.close()
        print("🔒 数据库连接已关闭！")