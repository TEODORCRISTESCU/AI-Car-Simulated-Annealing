# Research Reflection: Assessing Simulated Annealing in Autonomous Navigation

## Project Scope and Implementation Constraints

For this investigation, we utilized an existing **PyGame** framework to handle the basic physics engine and rendering of the vehicle. However, the core of the project, the artificial intelligence logic, was built entirely from scratch. We deliberately chose not to rely on pre-made AI libraries to fully test and experiment with **Simulated Annealing (SA)** and its training.

Finding a direct comparison for this specific implementation proved difficult. Most open-source projects for autonomous driving rely heavily on **Reinforcement Learning** or **Genetic Algorithms (GA)** equipped with sensory data. We were unable to find a project that applied "blind" Simulated Annealing to this specific track layout in a way that allowed for a 1:1 control group comparison. Consequently, our analysis focuses on our struggles and unique architecture compared to standard industry benchmarks.

## The Implementation Struggle: A Segmented Approach

Applying Simulated Annealing to driving controls wasn't smooth from the start. The biggest roadblock was the **"Horizon Problem."** Since our logic handled the track in sections, the AI often focused too much on short-term wins. It would hit a checkpoint, but end up facing a wall, making the next section impossible to complete (which was further worsened by our checkpoints being too large, but we didn't realize this until much later). 

To fix this, we implemented several key changes:
* **Angle Scoring:** We adjusted the scoring system to prioritize the car's angle. This taught the agent to sacrifice a little speed now for better positioning later.
* **Cooling Balance:** We had to balance trying new things with being consistent. If the system "cooled" down too fast, the car played it too safe and drove slowly. If it stayed too active, it drove erratically and lost progress.
* **Mode Switching:** We solved the cooling issue by building a system that switches between three modes: *standard learning*, *fine-tuning*, and a *"Panic" mode* to get things moving if the car gets stuck.



## Comparative Analysis: Simulated Annealing vs. Genetic Mutation

While we could not find a direct Simulated Annealing project to use as a base, we referenced the [GeneticCars](https://github.com/MikeOfZen/GeneticCars) repository (MikeOfZen) as a benchmark for performance in similar 2D environments. This comparison highlighted a critical divergence in architecture.

* **GeneticCars (Benchmark):** This implementation uses a population-based Genetic Algorithm paired with **ray-casting sensors**. This allows those agents to "see" the track walls and make decisions based on immediate proximity data.
* **Our Implementation:** Our Simulated Annealing agent operates **blindly**. It does not react to the walls; it memorizes a sequence of inputs (e.g., "Frame 10: Turn Left"). It makes it less efficient as it needs to learn the specific geometry of each track.



## Results and Performance Discrepancies

In the final analysis, the Genetic Algorithm approach referenced in the "GeneticCars" repo generally outperformed our Simulated Annealing model in terms of adaptability and convergence speed. The reasons for this were the following:

1. **Sensory Input vs. Memorization:** The Genetic agents could generalize. If a car using ray-casts is placed slightly off-center, it sees the wall and corrects. Our SA agent, lacking sensors, relies entirely on a rigid script of actions. A single pixel of displacement at the start leads to a crash later in the run because the memorized sequence no longer matches the physical reality of the car's position.
2. **Population vs. Iteration:** The Genetic approach benefits from parallel processing, namely, spawning 50 cars at once, which increases the odds of finding a "lucky" mutation. Our SA model optimizes a single agent sequentially. While this reduced the computational load, it meant that escaping a local minimum took significantly longer.

## Conclusion

The project successfully demonstrated that Simulated Annealing can solve complex navigation tasks, but it is not the optimal tool for dynamic environments. The final behavior of our agent was characterized by "jagged" movement—a byproduct of stochastic guessing rather than smooth, sensor-based decision-making. 

While the implementation was a success in terms of proving the concept, the comparison with sensor-based Genetic Algorithms confirms that for autonomous navigation, **"sight" (input data) is vastly superior to "memory" (action sequences).** The project, however, was a very good introduction to Machine Learning Algorithms, furthering our interest towards this topic and “forced” us to think of solutions to problems we didn’t think about before.

## References

* **MikeOfZen.** (n.d.). *GeneticCars: An Experiment/Game in machine learning.* GitHub. [https://github.com/MikeOfZen/GeneticCars](https://github.com/MikeOfZen/GeneticCars)
* **Techwithtim.** (n.d.). *Pygame-Car-Racer: Make a racing game in Python using pygame!* GitHub. [https://github.com/techwithtim/Pygame-Car-Racer](https://github.com/techwithtim/Pygame-Car-Racer)
* **GeeksforGeeks.** (2024, April 8). *Simulated Annealing.* GeeksforGeeks. [https://www.geeksforgeeks.org/dsa/simulated-annealing/](https://www.geeksforgeeks.org/dsa/simulated-annealing/)
