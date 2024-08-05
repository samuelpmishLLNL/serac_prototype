## Serac 0.x Problems: 

 

Format: problem statement (impact of problem) 

 

- no reverse mode in Functional (requires us to form sparse matrices, which is expensive) 

- FBar/Bbar doesn't really work (can’t analyze near-incompressible problems) 

- Functional has hardcoded "graph" (hard to modify code) 

- can't differentiate w.r.t. internal variables (plasticity models aren’t truly differentiable) 

- John Bowen's implementation of GPU kernels (initial RAJA implementation isn’t usable) 

- unreliable build (hurts productivity and creates bad user experience) 

- users are confused by dual number requirements (hurts user experience, counterintuitive) 

- inconsistent conventions for ownership, argument passing in physics modules (confusing to users) 

- physics modules don’t have clear goal, both legacy input-deck analyses and research tool (makes the design clumsy, since it’s being pulled in different directions) 

- state-machine physics module design makes it hard to understand/test (easier to introduce bugs and harder to fix them, also harder to do AD because of side effects in the state machine) 

 
...

 <div style="page-break-after: always; break-after: page;"></div>

## Serac 1.0 design goals: 


maximize (1.0 * performance - 0.1 * complexity) 

  such that 

the following requirements are satisfied: 

  - flexible 

  - intuitive 

  - differentiable (for now, only reverse-mode at "high" level) 

  - material support (especially with hysteresis) 

  - GPU implementation (by ~ Jan 2025) 

    - Hex8 Implicit Nonlinear Elasticity (medium-sized) problem 

  - reliable dependency resolution and build 

  

 

 

 

 

 <div style="page-break-after: always; break-after: page;"></div>

## Core tools: 

### Spatial Discretization (FE) 

 

See also: femto design doc 

 

 \- Meshes, Domains, and Fields 

 \- interpolation 

  \- can ask for different field quantities (value, differential operators, gradbar, ...) 

 \- integration 

  \- can integrate quantities against (against test functions, test function gradients, ...) 

 \- quadrature data handled already(?) 

 \- q-functions differentiated with Enzyme 

  \- risk mitigation: 

   \- fallback to finite difference 

   \- wrap use of Enzyme 

   \- commit to contributing to Enzyme 

   \- Enzyme is imminently going into LLVM (August-Sept 2024?) 

 

 

Mock Interface: 

 ```cpp
 Mesh mesh(“path/to/mesh”); 
 Domain domain(mesh, GaussLegendreRule(3)); 
 Field u(mesh, Family::CG, polynomial_order, components); 
 BasisFunction phi(u.function_space); 
 
 // basic residual calculation 
 auto du_dx_q = evaluate(grad(u), domain); 
 auto sigma_q = forall(material_model, du_dx_q); 
 Residual f = integrate(sigma_q * grad(phi), domain); 
 ```



<div style="page-break-after: always; break-after: page;"></div> 

### Time Integrator 

- explicit vjp 

- we'll own the implementation 

- don't care about exotic time integrators 

  

 

Mock Interface: 

 ```cpp
 // base class
 class TimeIntegrator{
     void advance(SimulationState & s, dt); // modify in-place or return?
     void vjp(...);
 }; 
 ```

 <div style="page-break-after: always; break-after: page;"></div>

### Solving Equations 

  

 \- explicit vjp 

 \- we'll own the implementation

- gets a residual function + list of constraints

- don't assume constraint list will be constant throughout time
  - can internally track constraints from previous invocation as performance optimization

 



 

### Constraints

- explicit vjp 

- linear / nonlinear 

- inequality / equality 

- special constraints / shortcuts (rigid body, single dof/node, box, positive) 

- support parametrizing constraints (see `Material Models`)
  - what should be allowed as a parameter? (from most to least important)
    - uniform variables (constant scalars/vectors/matrices over all quadrature points)
    - FE-interpolated fields
    - everything (eventually), but start with simpler set (constants in space, maybe FEM interpolations)? 

 

### Graph

 

 \- (sam) nice to have, not a requirement initially 

 \- (mike) invest in early, because it's helpful everywhere 

 \- Graphs containing subgraphs (nice to have) 

 \- Node requirements: 

  \- Primal, VJP 

  \- Who owns data (specifically output data) 

 \- Enzyme integration 

 

> Should the VJP get the "outputs" from the primal pass or not?
>
> - by default: nodes only get their primal inputs on reverse pass, but can opt-in to getting the primal outputs as well

  

 <div style="page-break-after: always; break-after: page;"></div>

### Material Models / Tractions / Body Forces

- needs to be "monolithic"

  - Don't encourage weird ad hoc coupling between dissimilar physics models
    - e.g. can't combine thermal material A with mechanical material B

- Support parametrizing tractions and body forces

  - Don't do:
    ```cpp
    // a, b passed implicitly, how to get sensitivities on graph?
    [a, b](double t, vec3 x) {
        return a * sin(x[0] + t) + b;
    }
    ```

    Instead do:

    ```cpp
    // a, b passed explicitly, appears on graph
    [](double t, vec3 x, double a, double b) {
        return a * sin(x[0] + t) + b;
    }
    ```

 <div style="page-break-after: always; break-after: page;"></div>

### Build System / Software Vending


TODO
 


 <div style="page-break-after: always; break-after: page;"></div>

## User's Input file

```cpp 
/// dynamics, materials with and without state
Field u{}, dudt{};

J2 mat1{};
// J2::internal_variables ?

Neohookean mat2{};

Mesh mesh = load("file.mesh");
Domain d1(mesh, "part1");
Domain d2(mesh, "part2");

Domain b1(mesh, "bdr1");
Domain b2(mesh, "bdr2");

/*
ResidualFunction supports:
- evaluation  (r(t, u, dudt; parameters; internal_variables?))
- vjp
- jvp

`ResidualFunction` is a temporary name
*/
ResidualFunction r(
  MaterialBlocks{{mat1, d1}, {mat2, d2}},
  Tractions{{traction, b1}},
  BodyForce{{gravity, part1}}    
  /* Note: no essential bcs here! */
);

Constraint essential_bcs{
    [](vec3 x) { return {0.0, 0.0, 1.0}; }, // fixity (one component, whole vector, ...)
    b2
};

Constraint inclined_plane_bcs = Constraint::inclined_plane(surface_normal, b2);

double dt = 1.0;
for (double t = 0.0; t < 10.0; t += dt) {
    
    
}
```



 
