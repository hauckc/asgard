#pragma once
#include "pde_base.hpp"

namespace asgard
{
// Example PDE using the 1D Diffusion Equation. This example PDE is
// time dependent (although not all the terms are time dependent). This
// implies the need for an initial condition.
// PDE: df/dt = d^2 f/dx^2

template<typename P>
class PDE_fokkerplanck_RF1 : public PDE<P>
{
public:
  PDE_fokkerplanck_RF1(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {
    v_thermal = 1; // executed at runtime, can use std::sqrt and other such methods
    nu_0 = electron_density * electron_density * electron_density * electron_density
      * coulomb_logarithm / (2 * M_PI * electron_density * v_thermal * v_thermal * v_thermal);
  }

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 1;
  static int constexpr num_terms_          = 3;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  static P constexpr nu_old = M_PI_2;   // TODO:  remove later
  static P constexpr diff_rf_0 = 1.0e6;  
  static P constexpr v_lower = 1.0;
  static P constexpr v_upper = 2.0;
  static P constexpr electron_plasma_frequency = 1.0;
  static P constexpr coulomb_logarithm =  1.0;
  static P constexpr electron_density = 1.0;
  static P constexpr T_e = 1.0;
  static P constexpr m_e = 1.0;
  static P constexpr delta = 1.0e-6;  // regularization parameter for collision frequency
  static P v_thermal;
  static P nu_0;
  
  static P constexpr z_eff = 1.0;   //TODO get this formula
  static P constexpr F_e0 = 1.0;    // This is the value of f_e at v = 0


  // TODO: figure out the values for the parameters above  

  // model parameters

  static P hat_diff_rf(P const v)
  {
    
    return ((v - v_lower)*(v - v_upper) < 0.0) ? diff_rf_0 : 0.0;
    
  }

   static P diff_rf(P const v, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    
    return hat_diff_rf(v) * (2.0  + z_eff)/2.0 * nu_0 * v_thermal * v_thermal;
    
  }

  static P collision_frequency(P const v, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    return nu_0 * std::pow(v_thermal / (std::abs(v) + delta),3);
  }

  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const &x, P const time = 0)
  {
    ignore(time);
    fk::vector<P> fx(x.size());
    // std::transform(x.begin(), x.end(), fx.begin(),
    //                [](P const &x_v) { return std::cos(nu_old * x_v); });
    for (int i =0; i < x.size(); i++)
    {
      fx(i) = F_e0 * std::exp(-x(i)*x(i)/v_thermal/v_thermal);
    }
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(0, 1, 3, 2, initial_condition_dim0, nullptr, "x");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0};

  /* Define terms */
  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::div, nullptr, nullptr, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::neumann);

  static fk::vector<P> bc_func(fk::vector<P> const x, P const t)
  {
    ignore(t);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const x_v) -> P { return std::cos(nu_old * x_v); });

    return fx;
  }

  static P bc_time_func(P const t)
  {
    /* e^(-2 * nu_old^2 * t )*/
    static double const p = -2.0 * nu_old * nu_old;
    return std::exp(p * t);
  }

  // TODO: Add interior penalty terms?
  // TODO: update nu_old value, check initial conditions
  inline static const partial_term<P> partial_term_1 = partial_term<P>(
      coefficient_type::grad, nullptr, nullptr, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, {bc_func},
      bc_time_func, {bc_func}, bc_time_func);

  inline static term<P> const term_0 =
      term<P>(true, // time-dependent
              "",   // name
              {partial_term_0, partial_term_1});

  // Interior penalty terms
  static P constexpr penalty = 0.0;
  static P g3(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return penalty;
  }
  static P g4(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return penalty;
  }
  inline static const partial_term<P> partial_term_2 = partial_term<P>(
      coefficient_type::div, g3, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, nullptr);

  inline static term<P> const term_1 = term<P>(false, // time-dependent
                                               "",    // name
                                               {partial_term_2});

  inline static const partial_term<P> partial_term_3 = partial_term<P>(
      coefficient_type::div, g4, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous, {}, nullptr, {},
      nullptr, nullptr);

  inline static term<P> const term_2 = term<P>(false, // time-dependent
                                               "",    // name
                                               {partial_term_3});

  inline static std::vector<term<P>> const terms_0 = {term_0};
  inline static std::vector<term<P>> const terms_1 = {term_1};
  inline static std::vector<term<P>> const terms_2 = {term_2};
  inline static term_set<P> const terms_ = {terms_0, terms_1, terms_2};

  /* Create sources */
  static fk::vector<P> source_0_x(fk::vector<P> const x, P const t)
  {
    ignore(t);
    static double const coefficient = -1.0 * nu_old * nu_old;

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return coefficient * std::cos(nu_old * x_v);
    });

    return fx;
  }

  static P source_0_t(P const t)
  {
    static double const coefficient = -2.0 * nu_old * nu_old;

    return std::exp(coefficient * t);
  }

  inline static source<P> const source_0 = source<P>({source_0_x}, source_0_t);

  inline static std::vector<source<P>> const sources_ = {source_0};

  /* exact solutions */
  static fk::vector<P> exact_solution_0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(nu_old * x_v); });
    return fx;
  }

  static fk::vector<P> exact_time(fk::vector<P> x, P const time)
  {
    x.resize(1);
    x[0] = source_0_t(time);
    return x;
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_0, exact_time};

  /* This is not used ever */
  inline static scalar_func<P> const exact_scalar_func_ = source_0_t;

  static P get_dt_(dimension<P> const &dim)
  {
    /* (1/2^level)^2 = 1/4^level */
    /* return dx; this will be scaled by CFL from command line */
    return std::pow(0.25, dim.get_level());
  }
};

template<typename P>
P PDE_fokkerplanck_RF1<P>::v_thermal;

template<typename P>
P PDE_fokkerplanck_RF1<P>::nu_0;

} // namespace asgard
