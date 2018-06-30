#include <cmath>
#include <iostream>
#include <stan/math.hpp>
#include <vector>

#include <Eigen/Dense>
#include <boost/math/tools/promotion.hpp>


namespace test
{

/*
 * boost::math::tools::promote_args<T1,...,TN>::type
 * T1,..,TNは数値型であり,その中で最も大きな小数型になる.(最低でもfloat)
 * 中にユーザー定義型が含まれていればそれになる
 * ユーザー定義型が複数ある場合,可換でなければならない.
 */
template <class T1, class T2, class T3>
inline typename boost::math::tools::promote_args<T1, T2, T3>::type
normal_log(T1 const& y, T2 const& mu, T3 const& sigma)
{
    using std::log;
    using std::pow;
    return -0.5 * pow((y - mu) / sigma, 2.0)
           - log(sigma)
           - 0.5 * log(2 * stan::math::pi());
};

using Eigen::Dynamic;
using Eigen::Matrix;

struct normal_ll {
    const Matrix<double, Dynamic, 1> y_;

    normal_ll(Matrix<double, Dynamic, 1> const& y) : y_(y){};

    template <class T>
    T operator()(Matrix<T, Dynamic, 1> const& theta) const
    {
        T mu = theta[0];
        T sigma = theta[1];
        T lp = 0;
        for (int n = 0; n < y_.size(); ++n) {
            lp += test::normal_log(y_[n], mu, sigma);
        }
        return lp;
    }
};

// (r,theta) -> (x,y)
struct rtheta_xy {
    template <class T>
    auto operator()(Eigen::Matrix<T, Dynamic, 1> rect) const
    {
        using std::cos;
        using std::sin;
        Eigen::Matrix<T, Dynamic, 1> ret(2);
        ret << rect(0) * cos(rect(1)), rect(0) * sin(rect(1));
        return ret;
    }
};

// (r,theta, phi) -> (x,y,z)
struct rthetaphi_xyz {
    template <class T>
    auto operator()(Eigen::Matrix<T, Dynamic, 1> pol) const
    {
        using std::cos;
        using std::sin;
        Eigen::Matrix<T, Dynamic, 1> ret(3);
        ret << pol(0) * sin(pol(1)) * cos(pol(2)), pol(0) * sin(pol(1)) * sin(pol(2)), pol(0) * cos(pol(1));
        return ret;
    }
};
}  // namespace test

int main()
{
    using stan::math::var;

    /*
    {
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
    }
    {
        using Eigen::Dynamic;
        using Eigen::Matrix;
        Matrix<double, Dynamic, 1> y(3);
        //y << 1.3, 2.7, -1.9;
        y << 1, 2, 3;
        test::normal_ll f{y};
        Matrix<double, Dynamic, 1> x(2);
        x << 1, 2;

        double fx;
        Matrix<double, Dynamic, 1> grad_fx;
        stan::math::gradient(f, x, fx, grad_fx);
        std::cout << "f(x) = " << fx << std::endl;
        std::cout << "grad f(x) = \n[" << grad_fx << "]" << std::endl;
    }
     */
    /*
    {

        using Eigen::Dynamic;
        using Eigen::Matrix;
        Matrix<double, Dynamic, 1> x(2);
        x << 2, stan::math::pi() / 4;
        std::cout << "f: (r,theta) -> (x,y)" << std::endl;
        test::rtheta_xy f{};
        std::cout << "f_x = \n[" << f(x) << "]" << std::endl;

        Matrix<double, Dynamic, Dynamic> J;
        Matrix<double, Dynamic, 1> f_x;
        // f.operator()はconst指定が必要
        stan::math::jacobian(f, x, f_x, J);
        std::cout << "x = [\n"
                  << x << "]" << std::endl;
        std::cout << "f_x = [\n"
                  << f_x << "]" << std::endl;
        std::cout << "J = [\n"
                  << J << "]" << std::endl;
    }
     */
    /*
    {

        using Eigen::Dynamic;
        using Eigen::Matrix;
        Matrix<double, Dynamic, 1> x(3);
        x << 2, stan::math::pi() / 4, stan::math::pi() / 4;
        std::cout << "f: (r,theta, phi) -> (x,y,z)" << std::endl;
        test::rthetaphi_xyz f{};
        std::cout << "f_x = \n[" << f(x) << "]" << std::endl;

        Matrix<double, Dynamic, Dynamic> J;
        Matrix<double, Dynamic, 1> f_x;
        // f.operator()はconst指定が必要
        stan::math::jacobian(f, x, f_x, J);
        std::cout << "x = [\n"
                  << x << "]" << std::endl;
        std::cout << "f_x = [\n"
                  << f_x << "]" << std::endl;
        std::cout << "J = [\n"
                  << J << "]" << std::endl;
    }
     */

    {
        std::vector<stan::math::var> vec{1, 2, 3};
        std::cout << stan::math::log_sum_exp(vec) << std::endl;
    }
    return 0;
}
