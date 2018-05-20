#include <iostream>
#include <cmath>
#include <stan/math.hpp>

#include <boost/math/tools/promotion.hpp>

namespace test {

template<class T1, class T2, class T3>
inline typename boost::math::tools::promote_args<T1, T2, T3>::type
normal_log(T1 const &y, T2 const &mu, T3 const &sigma) {
    using std::pow;
    using std::log;
    return -0.5 * pow((y - mu) / sigma, 2.0)
           - log(sigma)
           - 0.5 * log(2 * stan::math::pi());
};
}

int main() {
    double y = 1.3;
    stan::math::var mu = 0.5, sigma = 1.2;
    auto lp = test::normal_log(y, mu, sigma);
    std::vector<stan::math::var> theta;
    theta.push_back(mu);
    theta.push_back(sigma);
    std::vector<double> g;
    lp.grad(theta, g);
    std::cout << " d.f / d.mu = " << g[0]
              << " d.f / d.sigma = " << g[1] << std::endl;

    return 0;

}
