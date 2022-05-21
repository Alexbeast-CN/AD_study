from pyvirtualdisplay import Display
import gym
import deepq
import platform
import argparse

def load_actions ( action_filename ):

    actions = []

    with open ( action_filename ) as f:

        lines = f.readlines()

        for line in lines:
            action = []
            for tok in line.split():
                action.append ( float ( tok ))
            actions.append (action)

    return actions

def main():

    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 

    print ("python version:\t{0}".format (platform.python_version()))
    print ("gym version:\t{0}".format(gym.__version__))
    
    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument ( '--action_filename', type=str, default = 'default_actions.txt', help='a list of actions' )
    parser.add_argument ( '--cluster', default=False, action="store_true", help='a flag indicating whether training runs in the cluster' )
    parser.add_argument ( '--agent_name', type=str, default='agent')

    args = parser.parse_args()

    if args.cluster:
        display = Display ( visible = 0, size = ( 800, 600 ) )
        display.start ()

    # load actions
    actions = load_actions ( args.action_filename ) 
    print ( "actions:\t\t", actions )

    filename = args.agent_name +'.t7'
    print("loading {0}".format( filename ) )
    env = gym.make("CarRacing-v0")
    deepq.evaluate(env, new_actions = actions, load_path = filename )

    env.close()
    if args.cluster:
        display.stop ()


if __name__ == '__main__':
    main()
