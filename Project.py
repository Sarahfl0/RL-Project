import pygame
import numpy as np
import pickle
import pyautogui
import os,os.path
import matplotlib.pyplot as plt
import shutil

import time

from PIL import Image
import cv2
import pyscreenshot as ImageGab

pygame.init()

pink5 = (250, 100, 150)
pink6 = (250, 120, 180)
pink7 = (250, 140, 210)
pink8 = (250, 160, 240)
skyblue = (37, 207, 240)
white = (255, 255, 255)
red = (150, 0, 0)
ltred = (200, 0, 0)
black = (0, 0, 0)
green = (0, 155, 0)
blue = (0, 0, 255)
yellow = (200, 200, 0)
turkish_blue = (79, 151, 163)

smallfont = pygame.font.SysFont("comicsansms", 25)
midfont = pygame.font.SysFont("comicsansms", 50)
largefont = pygame.font.SysFont("comicsansms", 80)

FPS = 50
SIZE = 5
box_size = 100
display_height = SIZE * box_size
display_width = SIZE * box_size
agent_width = 100
agent_height = 100


GOAL_reward = 100
HOlE_punishment = -100
LIVING_reward = -1
epsilon = 0.9
EPS_DECAY = 0.9998
LEARNING_RATE = 0.1
DISCOUNT = 0.95
start_q_table = None

SHOW_EVERY = 1000
N_EPISODES = 12000
np.random.seed(1465)
clock = pygame.time.Clock()
display = pygame.display.set_mode((display_height, display_width))
run = True
screen_width = 1368
screen_height = 766
pos_x = screen_width / 2 - display_width / 2

pos_y = screen_height - display_height
os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x, pos_y)
os.environ['SDL_VIDEO_CENTERED'] = '0'
# print("pos_x, pos_y", pos_x, pos_y)
current_directory = os.getcwd()
final_directory = os.path.join(current_directory,r'Faiza_Project_ScreenShots')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
path = final_directory


def text_objects(text, color, size):
    if size == "small":
        textSurface = smallfont.render(text, True, color)
    elif size == "medium":
        textSurface = midfont.render(text, True, color)
    elif size == "large":
        textSurface = largefont.render(text, True, color)
    return textSurface, textSurface.get_rect()


def message_to_screen(msg, color, y_displace=0, size="small"):
    textSurf, textRect = text_objects(msg, color, size)

    textRect.center = int(display_width / 2), int(display_height / 2) + y_displace
    display.blit(textSurf, textRect)


if start_q_table is None:
    q_table = np.zeros((SIZE, SIZE, 4))
    for x1 in range(0, SIZE):
        for y1 in range(0, SIZE):
            for z1 in range(0, 4):
                q_table[x1][y1][z1] = np.random.uniform(-5, 0)

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    #size: size of each side of the grid
    #p:  decreasing it will add more holes to the environment. The proportion of normal states parameter
    """
    global res
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1 - p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


def rewardAndTerminal(row, column):
    terminal = False
    if map[row][column] == 'H':
        # print("You fell through a hole")
        r = HOlE_punishment
        terminal = True
    elif map[row][column] == 'G':
        # print("you reached the goal")
        r = GOAL_reward
        terminal = True
    else:
        r = LIVING_reward
    return r, terminal


class Agent():
    def __init__(self, x=None, y=None):
        if x == None:
            self.x = np.random.randint(0, SIZE)
        else:
            self.x = x
        if y == None:
            self.y = np.random.randint(0, SIZE)
        else:
            self.y = y

    def move(self, move_x=False, move_y=False):
        prev_x = self.x
        prev_y = self.y
        new_x = self.x + move_x
        new_y = self.y + move_y

        if new_x >= SIZE:
            self.x = SIZE - 1

        elif new_x < 0:
            self.x = 0
        else:
            self.x = new_x

        if new_y >= SIZE:
            self.y = SIZE - 1

        elif new_y < 0:
            self.y = 0
        else:
            self.y = new_y

        # print("prev_x,prev_y", prev_x, prev_y)
        pygame.draw.rect(display, colors_of_tiles[map[prev_y][prev_x]],
                         (prev_x * 100, prev_y * 100, agent_height, agent_width))
        # if(move_x==1 and move_y==0):
        #     print("self.x , self.y, agent_height, agent_width", self.x, self.y, agent_height, agent_width)
        #     print("new_x new_y move_x move_y",new_x,new_y,move_x,move_y)
        self.draw(white)

    def draw(self, color):
        pygame.draw.rect(display, color, (self.x * 100, self.y * 100, agent_height, agent_width))


map = generate_random_map(SIZE)
print(map)
n_states = SIZE * SIZE
n_actions = 4
actions=['Left','Down','Right','Up']

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
colors_of_tiles = {
    'F': skyblue,
    'S': turkish_blue,
    'H': red,
    'G': green
}

# clock.tick(3000000)

moves = 0
ima = []
ima1 = []


def inc(row, col, action):
    # print("**This block is working")
    if action == 0:
        col = max(col - 1, 0)
        a.move(-1, 0)
    elif action == 1:
        row = min(row + 1, SIZE - 1)
        a.move(0, 1)
    elif action == 2:
        col = min(col + 1, SIZE - 1)
        a.move(1, 0)
    elif action == 3:
        row = max(row - 1, 0)
        a.move(0, -1)

    return (row, col)
episode_rewards = []

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            quit()
    for episode in range(N_EPISODES+1):
        moves = 0
        if episode % SHOW_EVERY == 0:
            print(f"on # {episode}, epsilon: {epsilon}")
            # print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        reward_of_this_episode = 0
        done = False
        current_state = (0, 0)
        display.fill(skyblue)
        gridColor = black
        # drawing gridlines
        for i in range(0, SIZE):
            for j in range(0, SIZE):
                pygame.draw.rect(display, colors_of_tiles[map[i][j]], (j * box_size, i * box_size, box_size, box_size))
        a = Agent(current_state[0], current_state[1])
        if show:
            a.draw(color=white)
            pygame.display.update()
            # im1 = pyautogui.screenshot(region=(pos_x, pos_y, pos_x + 300, pos_y + 300))
            ima.append(pyautogui.screenshot())
            ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_0.png")))

        # if show:
        #     message_to_screen(f"Episode: {episode}",
        #                       black,
        #                       -30)
        #     pygame.display.update()
        #     ima.append(pyautogui.screenshot())
        #     ima1.append(pyautogui.screenshot(f"action_e_{episode}_m_{moves}.png"))
        while moves < 20:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[current_state[0]][current_state[1]])
            else:
                action = np.random.randint(0, 4)
            # print("currentstate:", current_state)
            # print("***action*****", action,"Action Name:",actions[action],"at move",moves)
            current_q = q_table[current_state[0]][current_state[1]][action]
            past_state = current_state
            current_state = inc(current_state[0], current_state[1], action)
            # print("new_state:", current_state)
            max_future_q = np.max(q_table[current_state[0]][current_state[1]])
            reward, terminal = rewardAndTerminal(current_state[0], current_state[1])
            # print("reward, terminal", reward, terminal)
            if terminal:
                new_q = reward
                done = True
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[past_state[0]][past_state[1]][action] = new_q
            # print("q_table at moves:", moves, q_table)

            # clock.tick(3000000)
            moves = moves + 1
            # clock.tick(FPS * 100000)
            if show:
                pygame.display.update()
                ima.append(pyautogui.screenshot())
                ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_{moves}.png")))
            # if video:
            #     next(save_screen)
            reward_of_this_episode += reward
            clock.tick(FPS * 100000)
            if done == True:
                if show:
                    pygame.display.update()
                    ima.append(pyautogui.screenshot())
                    ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_{moves}.png")))
                    moves = moves + 1
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                    display.fill(white)
                    if reward == HOlE_punishment:
                        # print("You fell through a hole")
                        message_to_screen(f"Episode{episode}",
                                          pink5,
                                          -100)
                        message_to_screen("Fell through a hole",
                                          red,
                                          -50,
                                          "small")

                    elif reward == GOAL_reward:
                        # print("You reached the goal")
                        message_to_screen(f"Episode{episode}",
                                          pink5,
                                          -100)
                        message_to_screen(f"Reached the goal in {moves-2} moves",
                                          green,
                                          -50)

                    pygame.display.update()
                    ima.append(pyautogui.screenshot())
                    ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_{moves}.png")))
                    # if video:
                    #     next(save_screen)
                    clock.tick(40000)
                    moves = 20
                    # run = False
                    pygame.display.update()
                    clock.tick(1500)
        if done == False:
            if show:
                moves = moves + 1
                pygame.display.update()

                ima.append(pyautogui.screenshot())
                ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_{moves}.png")))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                display.fill(white)
                # print("20 moves but no result")
                message_to_screen(f"Episode{episode}",
                                  pink5,
                                  -100)
                message_to_screen("20 steps completed but no result",
                                  black,
                                  -50)
                pygame.display.update()
                ima.append(pyautogui.screenshot())
                ima1.append(pyautogui.screenshot(os.path.join(path, f"action_e_{episode}_m_{moves}.png")))
                # if video:
                #     next(save_screen)
                clock.tick(40000)
                moves = 20
                pygame.display.update()
                clock.tick(1500)
        if show:
            print(f"reward of this episode{reward_of_this_episode} ")
        episode_rewards.append(reward_of_this_episode)
        epsilon *= EPS_DECAY

    run = False
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{N_EPISODES}.pickle", "wb") as f:
    pickle.dump(q_table, f)
shutil.rmtree(final_directory)
