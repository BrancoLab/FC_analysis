import datajoint as dj



dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'tutorial'


dj.conn()




schema = dj.schema('tutorial', locals())


@schema
class Mouse(dj.Manual):
      definition = """
      mouse_id: int                  # unique mouse id
      ---
      dob: date                      # mouse date of birth
      sex: enum('M', 'F', 'U')    # sex of mouse - Male, Female, or Unknown/Unclassified"""




mouse = Mouse()

print(mouse)


# data = [
#   (1, '2016-11-19', 'M'),
#   (2, '2016-11-20', 'U'),
#   (5, '2016-12-25', 'F')
# ]
#
# # now insert all at once
# mouse.insert(data)
#
# print(mouse)



















