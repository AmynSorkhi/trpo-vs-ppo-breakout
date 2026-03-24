# TRPO Folder

If looking to clone results, use the "Updated Files" Folder.


### "From Scratch" Implementation Design [original]

1. policy network (A-C, shared CNN backbone)
2. trajectory collection (gather experience tuples - s,a,r,s',)
3. advantage calculation (GAE, advantage normalization)
4. surrogate objective (measure policy improvement)
5. Fisher Information Matrix (FIM)
6. Conjugate Gradient Solver
7. Line Search
8. Value function optimization
9. KL Divergence calculator

### Environment 

Atari Breakout V4/V5
