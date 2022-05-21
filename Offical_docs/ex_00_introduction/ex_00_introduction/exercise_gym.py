from pyvirtualdisplay import Display
import gym

display = Display(visible=0, size=(800,600))
display.start()
env = gym.make('CarRacing-v0')
obs, done = env.reset(), False
ep_rew = 0.0
while not done:
   obs, rew, done, _ = env.step([0.0, 1.0, 0.0])
   ep_rew += rew
print(ep_rew)
display.stop()
