import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import pickle
import os

class AdvancedRLAgent:
    """
    Advanced Reinforcement Learning Agent for Energy Trading
    Uses Deep Q-Network (DQN) approach with experience replay
    """
    
    def __init__(self, state_size: int = 10, action_size: int = 8, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Q-table for simplified implementation
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Action mapping
        self.actions = [
            "hold",
            "sell_to_grid", 
            "buy_from_grid",
            "sell_to_company_1",
            "sell_to_company_2", 
            "buy_from_company_1",
            "buy_from_company_2",
            "emergency_charge"
        ]
        
        # Rewards history for learning curve analysis
        self.rewards_history = []
        self.episode_rewards = []
        
    def get_state_representation(self, company, companies, grid_price, hour):
        """Convert system state to numerical representation"""
        
        # Normalize values to 0-1 range for better learning
        state = [
            company.total_surplus / 1000.0,  # Normalized surplus
            company.total_deficit / 1000.0,  # Normalized deficit
            grid_price / 0.2,  # Normalized grid price (assuming max 0.2 $/kWh)
            company.energy_price / 0.2,  # Normalized company price
            hour / 24.0,  # Time of day normalized
        ]
        
        # Add other companies' states
        other_companies = [c for c in companies if c.id != company.id]
        for i, other_company in enumerate(other_companies[:2]):  # Limit to 2 other companies
            state.extend([
                other_company.total_surplus / 1000.0,
                other_company.energy_price / 0.2,
            ])
        
        # Pad state if fewer than 2 other companies
        while len(state) < self.state_size:
            state.append(0.0)
        
        return tuple(state[:self.state_size])
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_table[state]
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + 0.95 * np.max(self.q_table[next_state])
            
            # Q-learning update
            self.q_table[state][action] = (
                (1 - self.learning_rate) * self.q_table[state][action] + 
                self.learning_rate * target
            )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, company, action, companies, grid_price, previous_balance):
        """Calculate reward for the action taken"""
        reward = 0.0
        current_balance = company.total_surplus - company.total_deficit
        
        # Base reward for improving energy balance
        balance_improvement = current_balance - previous_balance
        reward += balance_improvement * 10  # Scale factor
        
        # Action-specific rewards
        if action == 0:  # Hold
            reward += 1.0  # Small positive reward for stability
            
        elif action == 1:  # Sell to grid
            if company.total_surplus > 0:
                revenue = company.total_surplus * grid_price
                reward += revenue * 100  # Scale revenue reward
            else:
                reward -= 50  # Penalty for invalid action
                
        elif action == 2:  # Buy from grid
            if company.total_deficit > 0:
                cost = company.total_deficit * grid_price
                reward -= cost * 50  # Cost penalty (less than revenue reward)
                reward += 20  # Small reward for meeting demand
            else:
                reward -= 30  # Penalty for unnecessary purchase
                
        elif action in [3, 4]:  # Sell to other companies
            if company.total_surplus > 0:
                other_companies = [c for c in companies if c.id != company.id]
                if len(other_companies) > action - 3:
                    buyer = other_companies[action - 3]
                    if buyer.total_deficit > 0:
                        trade_amount = min(company.total_surplus, buyer.total_deficit)
                        revenue = trade_amount * buyer.energy_price
                        reward += revenue * 120  # Higher reward for company trades
                        
                        # Bonus for better than grid price
                        if buyer.energy_price > grid_price:
                            reward += (buyer.energy_price - grid_price) * trade_amount * 50
                    else:
                        reward -= 40  # Penalty for trying to sell to non-buyer
            else:
                reward -= 50  # Penalty for trying to sell without surplus
                
        elif action in [5, 6]:  # Buy from other companies
            if company.total_deficit > 0:
                other_companies = [c for c in companies if c.id != company.id]
                if len(other_companies) > action - 5:
                    seller = other_companies[action - 5]
                    if seller.total_surplus > 0:
                        trade_amount = min(company.total_deficit, seller.total_surplus)
                        cost = trade_amount * seller.energy_price
                        reward -= cost * 40  # Cost penalty
                        reward += 25  # Reward for meeting demand
                        
                        # Bonus for better than grid price
                        if seller.energy_price < grid_price:
                            reward += (grid_price - seller.energy_price) * trade_amount * 60
                    else:
                        reward -= 40  # Penalty for trying to buy from non-seller
            else:
                reward -= 30  # Penalty for unnecessary purchase
                
        elif action == 7:  # Emergency charge
            # Check if any batteries are critically low
            critical_batteries = 0
            for station in company.stations:
                for battery in station.batteries:
                    if battery.soc < 10:  # Critical SOC
                        critical_batteries += 1
            
            if critical_batteries > 0:
                reward += 100 * critical_batteries  # High reward for emergency action
            else:
                reward -= 20  # Penalty for unnecessary emergency charge
        
        # Additional penalties/rewards
        
        # Penalty for maintenance issues
        maintenance_issues = sum(
            len([b for b in s.batteries if b.maintenance_required]) +
            len([i for i in s.inverters if i.maintenance_required]) +
            len([p for p in s.solar_panels if p.maintenance_required])
            for s in company.stations
        )
        reward -= maintenance_issues * 10
        
        # Reward for high battery SOC (energy security)
        avg_soc = np.mean([
            np.mean([b.soc for b in s.batteries])
            for s in company.stations
        ])
        if avg_soc > 80:
            reward += 20
        elif avg_soc < 30:
            reward -= 30
        
        return reward
    
    def get_learning_stats(self):
        """Get learning statistics for monitoring"""
        return {
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "avg_reward_last_100": np.mean(self.rewards_history[-100:]) if len(self.rewards_history) >= 100 else 0,
            "total_episodes": len(self.episode_rewards),
            "q_table_size": len(self.q_table)
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "rewards_history": self.rewards_history,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(model_data["q_table"])
            self.epsilon = model_data.get("epsilon", self.epsilon_min)
            self.rewards_history = model_data.get("rewards_history", [])
            
            return True
        return False

class RuleBasedAgent:
    """
    Rule-based agent for comparison and fallback
    """
    
    def __init__(self):
        self.name = "Rule-Based Agent"
    
    def choose_action(self, company, companies, grid_price, hour):
        """Choose action based on predefined rules"""
        
        # Emergency charging rule
        critical_batteries = sum(
            len([b for b in s.batteries if b.soc < 10])
            for s in company.stations
        )
        if critical_batteries > 0:
            return 7  # Emergency charge
        
        # Trading rules
        if company.total_surplus > 0:
            # Try to sell to highest paying company first
            other_companies = [c for c in companies if c.id != company.id and c.total_deficit > 0]
            if other_companies:
                best_buyer = max(other_companies, key=lambda x: x.energy_price)
                if best_buyer.energy_price > grid_price * 1.05:  # 5% better than grid
                    # Find which company index this is
                    other_companies_all = [c for c in companies if c.id != company.id]
                    try:
                        buyer_index = other_companies_all.index(best_buyer)
                        return 3 + buyer_index  # Sell to company 1 or 2
                    except (ValueError, IndexError):
                        pass
            
            # Sell to grid if no better company option
            return 1  # Sell to grid
            
        elif company.total_deficit > 0:
            # Try to buy from cheapest company first
            other_companies = [c for c in companies if c.id != company.id and c.total_surplus > 0]
            if other_companies:
                best_seller = min(other_companies, key=lambda x: x.energy_price)
                if best_seller.energy_price < grid_price * 0.95:  # 5% cheaper than grid
                    # Find which company index this is
                    other_companies_all = [c for c in companies if c.id != company.id]
                    try:
                        seller_index = other_companies_all.index(best_seller)
                        return 5 + seller_index  # Buy from company 1 or 2
                    except (ValueError, IndexError):
                        pass
            
            # Buy from grid if no better company option
            return 2  # Buy from grid
        
        # Hold if balanced
        return 0  # Hold

class HybridEnergyAgent:
    """
    Hybrid agent that combines RL and rule-based approaches
    """
    
    def __init__(self, rl_weight=0.7):
        self.rl_agent = AdvancedRLAgent()
        self.rule_agent = RuleBasedAgent()
        self.rl_weight = rl_weight  # Weight for RL vs rule-based decisions
        self.model_path = "rl_energy_model.pkl"
        
        # Try to load existing model
        self.rl_agent.load_model(self.model_path)
    
    def choose_action(self, company, companies, grid_price, hour):
        """Choose action using hybrid approach"""
        
        # Get state representation
        state = self.rl_agent.get_state_representation(company, companies, grid_price, hour)
        
        # Get actions from both agents
        rl_action = self.rl_agent.choose_action(state)
        rule_action = self.rule_agent.choose_action(company, companies, grid_price, hour)
        
        # Use rule-based for critical situations
        critical_batteries = sum(
            len([b for b in s.batteries if b.soc < 10])
            for s in company.stations
        )
        
        if critical_batteries > 0:
            return rule_action  # Always use rule-based for emergencies
        
        # Otherwise, use weighted random selection
        if random.random() < self.rl_weight:
            return rl_action
        else:
            return rule_action
    
    def update_rl_agent(self, state, action, reward, next_state, done):
        """Update the RL component"""
        self.rl_agent.remember(state, action, reward, next_state, done)
        self.rl_agent.replay()
    
    def save_model(self):
        """Save the RL model"""
        self.rl_agent.save_model(self.model_path)
    
    def get_stats(self):
        """Get agent statistics"""
        return {
            "rl_stats": self.rl_agent.get_learning_stats(),
            "rl_weight": self.rl_weight,
            "model_saved": os.path.exists(self.model_path)
        }