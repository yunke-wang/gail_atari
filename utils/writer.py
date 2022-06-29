import os


class Writer(object):
    def __init__(self, env, seed, prior, traj_size, fname='', folder='PU_log'):
        if fname != '':
            fname = '_{}'.format(fname)
        if prior > 1e-6:
            plabel = '_{:.2f}'.format(prior)
        else:
            plabel = ''

        self.fname = '{}_{}_{}_{}_{}.csv'.format(env, seed, traj_size, plabel, fname)
        self.folder = folder
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        if os.path.exists('{}/{}'.format(self.folder, self.fname)):
            print('Overwrite {}/{}!'.format(self.folder, self.fname))
            os.remove('{}/{}'.format(self.folder, self.fname))

    def log(self, epoch, reward,var):
        with open(self.folder + '/' + self.fname, 'a') as f:
            f.write('{},{},{}\n'.format(epoch, reward, var))
