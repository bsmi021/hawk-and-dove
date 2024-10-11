# main.py (located at d:\Projects\edsl_test\hawk-and-dove\src\main.py)

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from mesa.visualization import Slider, Choice, NumberInput
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
import sys
import numpy as np
import pandas as pd
import random

class HawkDoveAgent(Agent):
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.resources = 0
        self.age = 0
        self.interactions = {"Hawk": 0, "Dove": 0}
        self.wins = 0
        self.losses = 0
        self.alive = True
        self.dominance_score = 0
        self.max_resources = model.max_agent_resources

    def step(self):
        if not self.alive or self.pos is None:
            return
        self.move()
        self.interact_with_resource()
        self.interact()
        self.reproduce()
        self.age += 1
        self.metabolism()
        self.update_dominance_score()
        self.share_resources()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def interact_with_resource(self):
        cell_content = self.model.grid.get_cell_list_contents([self.pos])
        resources = [obj for obj in cell_content if isinstance(obj, Resource)]
        if resources:
            resource = self.random.choice(resources)
            gained_resources = min(resource.value, self.max_resources - self.resources)
            self.resources += gained_resources
            resource.value -= gained_resources
            if resource.value <= 0:
                self.model.grid.remove_agent(resource)
                self.model.resource_count -= 1

    def interact(self):
        cell_content = self.model.grid.get_cell_list_contents([self.pos])
        other_agents = [obj for obj in cell_content if isinstance(obj, HawkDoveAgent) and obj != self and obj.alive]
        if other_agents:
            other_agent = self.random.choice(other_agents)
            self.resolve_interaction(other_agent)

    def resolve_interaction(self, other_agent):
        self.interactions[other_agent.strategy] += 1
        other_agent.interactions[self.strategy] += 1

        if self.strategy == "Hawk" and other_agent.strategy == "Hawk":
            if self.random.random() < 0.5:
                gained_resources = min(self.model.resource_value / 2, self.max_resources - self.resources)
                self.resources += gained_resources
                other_agent.resources = max(0, other_agent.resources - self.model.injury_cost)
                self.wins += 1
                other_agent.losses += 1
            else:
                self.resources = max(0, self.resources - self.model.injury_cost)
                gained_resources = min(self.model.resource_value / 2, other_agent.max_resources - other_agent.resources)
                other_agent.resources += gained_resources
                self.losses += 1
                other_agent.wins += 1
        elif self.strategy == "Hawk" and other_agent.strategy == "Dove":
            gained_resources = min(self.model.resource_value, self.max_resources - self.resources)
            self.resources += gained_resources
            self.wins += 1
            other_agent.losses += 1
        elif self.strategy == "Dove" and other_agent.strategy == "Hawk":
            gained_resources = min(self.model.resource_value, other_agent.max_resources - other_agent.resources)
            other_agent.resources += gained_resources
            self.losses += 1
            other_agent.wins += 1
        elif self.strategy == "Dove" and other_agent.strategy == "Dove":
            shared_resource = self.model.resource_value / 2
            gained_resources = min(shared_resource, self.max_resources - self.resources)
            self.resources += gained_resources
            gained_resources_other = min(shared_resource, other_agent.max_resources - other_agent.resources)
            other_agent.resources += gained_resources_other

        self.model.interaction_outcomes[f"{self.strategy}-{other_agent.strategy}"] += 1

    def reproduce(self):
        if self.resources >= self.model.reproduction_threshold:
            current_density = len(self.model.schedule.agents) / (self.model.grid.width * self.model.grid.height)
            reproduction_chance = 1 - current_density
            if self.random.random() < reproduction_chance:
                offspring = HawkDoveAgent(self.model.next_id(), self.model, self.strategy)
                self.model.grid.place_agent(offspring, self.pos)
                self.model.schedule.add(offspring)
                self.resources -= self.model.reproduction_cost
                offspring.resources = self.model.reproduction_cost / 2

    def metabolism(self):
        base_metabolism_rate = 1 if self.strategy == "Dove" else 1.5  # Hawks have higher base metabolism
        metabolism_rate = base_metabolism_rate * (1 + self.resources / self.max_resources)  # Metabolism increases with fullness
        self.resources = max(0, self.resources - metabolism_rate)
        if self.resources <= 0 or self.age > self.model.max_age:
            self.die()

    def die(self):
        self.alive = False
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)
        if self.unique_id in self.model.agents:
            del self.model.agents[self.unique_id]

    def update_dominance_score(self):
        self.dominance_score = self.wins - self.losses

    def share_resources(self):
        if not self.alive or self.pos is None:
            return
        if self.strategy == "Dove" and self.resources > self.max_resources * 0.75:
            cell_content = self.model.grid.get_cell_list_contents([self.pos])
            other_doves = [obj for obj in cell_content if isinstance(obj, HawkDoveAgent) and obj != self and obj.strategy == "Dove" and obj.alive]
            if other_doves:
                share_amount = (self.resources - self.max_resources * 0.5) / (len(other_doves) + 1)
                for dove in other_doves:
                    transferred = min(share_amount, dove.max_resources - dove.resources)
                    dove.resources += transferred
                    self.resources -= transferred

class Resource(Agent):
    def __init__(self, unique_id, model, value):
        super().__init__(unique_id, model)
        self.value = value
    # ... [The rest of the HawkDoveAgent class remains unchanged]

class HawkDoveModel(Model):
    def __init__(
        self,
        N,
        width,
        height,
        resource_value,
        injury_cost,
        initial_hawk_ratio,
        reproduction_threshold,
        reproduction_cost,
        resource_density,
        max_agent_resources,
        seed=None,
    ):
        super().__init__()
        self.num_agents = int(N)
        self.grid = MultiGrid(int(width), int(height), torus=True)
        self.schedule = RandomActivation(self)
        self.resource_value = float(resource_value)
        self.injury_cost = float(injury_cost)
        self.initial_hawk_ratio = float(initial_hawk_ratio)
        self.reproduction_threshold = float(reproduction_threshold)
        self.reproduction_cost = float(reproduction_cost)
        self.resource_density = float(resource_density)
        self.max_agent_resources = float(max_agent_resources)
        self.running = True
        self.max_resources = int(self.grid.width * self.grid.height * self.resource_density * 2)
        self.resource_count = 0
        self.carrying_capacity = int(self.grid.width * self.grid.height * 0.75)  # 75% of grid size
        self.max_age = 100  # Maximum age for agents
        self.interaction_outcomes = {
            "Hawk-Hawk": 0,
            "Hawk-Dove": 0,
            "Dove-Hawk": 0,
            "Dove-Dove": 0
        }
        self.agents = {}

        # Set the random seed
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.random.seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        for i in range(self.num_agents):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            strategy = "Hawk" if self.random.random() < self.initial_hawk_ratio else "Dove"
            agent = HawkDoveAgent(self.next_id(), self, strategy)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
            self.agents[agent.unique_id] = agent

        self.initialize_resources()

        self.datacollector = DataCollector(
            model_reporters={
                "Hawk Count": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Hawk" and a.alive),
                "Dove Count": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Dove" and a.alive),
                "Average Hawk Resources": lambda m: np.mean([a.resources for a in m.schedule.agents if a.strategy == "Hawk" and a.alive]) if any(a.strategy == "Hawk" and a.alive for a in m.schedule.agents) else 0,
                "Average Dove Resources": lambda m: np.mean([a.resources for a in m.schedule.agents if a.strategy == "Dove" and a.alive]) if any(a.strategy == "Dove" and a.alive for a in m.schedule.agents) else 0,
                "Resource Count": lambda m: m.resource_count,
                "Hawk-Hawk Interactions": lambda m: m.interaction_outcomes["Hawk-Hawk"],
                "Hawk-Dove Interactions": lambda m: m.interaction_outcomes["Hawk-Dove"],
                "Dove-Dove Interactions": lambda m: m.interaction_outcomes["Dove-Dove"],
                "Seed": lambda m: m.seed,
            },
            agent_reporters={
                "Strategy": "strategy",
                "Resources": "resources",
                "Age": "age",
                "Hawk Interactions": lambda a: a.interactions["Hawk"],
                "Dove Interactions": lambda a: a.interactions["Dove"],
                "Wins": "wins",
                "Losses": "losses",
                "Alive": "alive",
                "Dominance Score": "dominance_score",
            }
        )

  
    def initialize_resources(self):
        for _ in range(int(self.grid.width * self.grid.height * self.resource_density)):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            resource = Resource(self.next_id(), self, self.resource_value)
            self.grid.place_agent(resource, (x, y))
            self.resource_count += 1

    def replenish_resources(self):
        resources_to_add = min(
            int(self.grid.width * self.grid.height * self.resource_density * 0.1),  # Replenish 10% of desired density
            self.max_resources - self.resource_count
        )

        for _ in range(resources_to_add):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            resource = Resource(self.next_id(), self, self.resource_value)
            self.grid.place_agent(resource, (x, y))
            self.resource_count += 1

    def step(self):
        self.schedule.step()
        self.replenish_resources()
        self.datacollector.collect(self)

class AgentDataTable(TextElement):
    def render(self, model):
        agent_data = []
        for agent in model.agents.values():
            agent_data.append({
                "ID": agent.unique_id,
                "Strategy": agent.strategy,
                "Resources": agent.resources,
                "Age": agent.age,
                "Hawk Interactions": agent.interactions["Hawk"],
                "Dove Interactions": agent.interactions["Dove"],
                "Wins": agent.wins,
                "Losses": agent.losses,
                "Dominance Score": agent.dominance_score,
                "Status": "Alive" if agent.alive else "Dead"
            })
        df = pd.DataFrame(agent_data)
        return df.to_html(index=False, classes=["table", "table-striped", "table-bordered"])

def agent_portrayal(agent):
    if isinstance(agent, HawkDoveAgent):
        portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 1}
        if agent.strategy == "Hawk":
            color_intensity = min(255, max(0, int(128 + agent.dominance_score * 10)))
            portrayal["Color"] = f"rgb({color_intensity},0,0)"
        else:
            portrayal["Color"] = "blue"
        portrayal["text"] = str(int(agent.resources))
        portrayal["text_color"] = "white"
    elif isinstance(agent, Resource):
        size = 0.1 + (agent.value / 20) * 0.4
        portrayal = {
            "Shape": "rect",
            "w": size,
            "h": size,
            "Filled": "true",
            "Color": "green",
            "Layer": 0,
        }
    else:
        portrayal = {}
    return portrayal
  
if __name__ == "__main__":
    params = {
        "N": Slider("Number of Agents", 100, 10, 200, 1),
        "width": 20,
        "height": 20,
        "resource_value": Slider("Resource Value", 10, 1, 50, 1),
        "injury_cost": Slider("Injury Cost", 20, 1, 50, 1),
        "initial_hawk_ratio": Slider("Initial Hawk Ratio", 0.5, 0, 1, 0.1),
        "reproduction_threshold": Slider("Reproduction Threshold", 30, 10, 100, 1),
        "reproduction_cost": Slider("Reproduction Cost", 20, 5, 50, 1),
        "resource_density": Slider("Resource Density", 0.3, 0, 1, 0.1),
        "max_agent_resources": Slider("Max Agent Resources", 50, 20, 100, 5),
        "seed": NumberInput("Random Seed", 42),
    }

    grid = CanvasGrid(agent_portrayal, params["width"], params["height"], 500, 500)

    population_chart = ChartModule([
        {"Label": "Hawk Count", "Color": "Red"},
        {"Label": "Dove Count", "Color": "Blue"}
    ])
    resource_chart = ChartModule([
        {"Label": "Average Hawk Resources", "Color": "Red"},
        {"Label": "Average Dove Resources", "Color": "Blue"},
        {"Label": "Resource Count", "Color": "Green"}
    ])
    interaction_chart = ChartModule([
        {"Label": "Hawk-Hawk Interactions", "Color": "Red"},
        {"Label": "Hawk-Dove Interactions", "Color": "Purple"},
        {"Label": "Dove-Dove Interactions", "Color": "Blue"}
    ])

    agent_data_table = AgentDataTable()

    server = ModularServer(
        HawkDoveModel,
        [grid, population_chart, resource_chart, interaction_chart, agent_data_table],
        "Hawk Dove Model",
        params,
    )
    server.port = 8521  # Default server port
    server.launch()
