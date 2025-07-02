import os
from datetime import datetime

if __name__ == '__main__':

    # os.system('python data_fix_diffH_snapshot.py')
    # print('@@@@@@@@@@@@@@@@@@End of data_fix_diffH_snapshot@@@@@@@@@@@@@@@@@@')

    start = datetime.now()
    os.system('python data_fix.py')
    print('@@@@@@@@@@@@@@@@@@End of data_fix@@@@@@@@@@@@@@@@@@')
    print("Total time:", datetime.now() - start)
    os.system('python data_fix_snapshot.py')
    print('@@@@@@@@@@@@@@@@@@End of data_fix_snapshot@@@@@@@@@@@@@@@@@@')
    print("Total time:", datetime.now() - start)
