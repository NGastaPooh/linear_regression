import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation


def main():
    # get the dataframe (df)
    df_baseline = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

    # chose initial input and output data series for the simple least squares fit
    df = df_baseline[['1stFlrSF', 'SalePrice']]

    # scale the data similar to sklearn MinMaxScaler()
    # necessary for Gradient Descent method to work
    # assumes that [0, 0] is a part of the dataset
    df_scaled = df/(df.max())

    # convert to numpy arrays
    x = (df_scaled['1stFlrSF'].to_numpy()).astype(float)
    y = (df_scaled['SalePrice'].to_numpy()).astype(float)

    # calculate analytical values for fitting coefficients; to be used as reference values later on
    # see details at https://en.wikipedia.org/wiki/Simple_linear_regression#Example
    x_avg = np.average(x)
    y_avg = np.average(y)

    beta_LQ = sum((x - x_avg) * (y - y_avg)) / sum((x - x_avg) ** 2)
    alpha_LQ = y_avg - beta_LQ * x_avg

    # add an extra column to input data to follow the convention, see lectures
    # specifically for 1D case, i write it in this form
    x1 = np.vstack((np.ones(len(x)), x)).T

    # initial guess for coefficients
    # could be any value; depend on the context and conditions of the problem
    # in current case, all the data is scaled to the interval [0, 1], so this seemed to be a reasonable choice
    # in general, it may affect the convergence time, as mentioned in lectures
    theta = np.array([0.5, 0.5])
    theta_1 = np.array([0.5, 0.5])

    # initialize some utility arrays
    partial_derivative = np.array([1., 1.])
    hessian = np.zeros((2, 2))

    # Gradient Descent coefficient
    # there are papers on how to choose it
    # here this value was picked by hand
    alpha = 0.01

    # Gradient Descent algorythm that does "grad_steps" iterations
    # error is chosen to be a distance from currently calculated values of fitting coefficients to
    # analytically derived ones
    grad_steps = 30
    grad_error = np.zeros(grad_steps)
    # store all iterations of theta to create an animation later on
    theta_grad_animation = np.zeros((grad_steps, 2))

    for k in range(grad_steps):
        for i in range(len(theta_1)):
            theta_1[i] = theta_1[i] - alpha * sum(x1[:, i] * (np.sum(theta_1 * x1, axis=1) - y))
            theta_grad_animation[k][i] = theta_1[i]
        grad_error[k] = abs(sum([ii ** 2 for ii in theta_1]) - alpha_LQ ** 2 - beta_LQ ** 2)

    # Newton's algorythm; notation is similar to Gradient Descent
    newton_steps = 30
    newton_error = np.zeros(newton_steps)
    theta_newton_animation = np.zeros((grad_steps, 2))

    for k in range(newton_steps):
        for i in range(len(theta)):
            partial_derivative[i] = sum(x1[:, i] * (np.sum(theta * x1, axis=1) - y))
            for j in range(len(theta)):
                hessian[i][j] = sum(x1[:, i] * x1[:, j])
        hessian_inv = np.linalg.inv(hessian)
        theta = theta - hessian_inv @ partial_derivative
        theta_newton_animation[k][:] = theta[:]
        newton_error[k] = abs(sum([ii**2 for ii in theta]) - alpha_LQ**2 - beta_LQ**2)

    # initialize matplotlib figure
    fig = plt.figure()
    ax = plt.gca()

    # visualize the 1st frame, set up color scheme
    sns.scatterplot(x="1stFlrSF", y="SalePrice", color="blue", data=df_scaled, ax=ax)
    ax.set_xlim(0, 1.02)  # should be a bit larger than the data range ([0, 1] in this case) for better visualization
    ax.set_ylim(0, 1.02)

    # draw the analytical best fit line as an eye guideline
    x_for_analytical_fit = np.linspace(0, 1.02, 100)
    ax.plot(x_for_analytical_fit, alpha_LQ + beta_LQ * x_for_analytical_fit, "--k", lw=0.7)

    # a function that generates n images of the system; needed for the animation
    def animate(n):
        y_gradient = theta_grad_animation[n][0] + theta_grad_animation[n][1] * x
        y_newton = theta_newton_animation[n][0] + theta_newton_animation[n][1] * x
        line_newton, = ax.plot(x, y_newton, 'sg', markersize=4)
        line_gradient, = ax.plot(x, y_gradient, '.r', markersize=6)
        return line_newton, line_gradient,

    # animate system evolution with 400 ms delay between frames
    animate = animation.FuncAnimation(fig, animate, frames=newton_steps,
                                      interval=400, blit=True, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()