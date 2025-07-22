import numpy as np


class StorageBox():
    def __init__(self, initial_storage, initial_mass,
                 reaction_rate=0, retardation_factor=1,
                 evapoconcentration_factor=1, dt=1,
                 run_NMBC_Q=False, run_NMBC_ET=False,
                 run_NMBC_react=False, run_water_bTTD=False,
                 run_tracer_bTTD=False, max_age_tracked=None):
        """
        A conceptual storage with reactive tracer processes including 
        sorption (via retardation), evapoconcentration, and first-order decay.

        Parameters
        ----------
        initial_storage : float
            Initial water volume in storage [L]
        initial_mass : float
            Initial mass of tracer in the storage [M]
        reaction_rate : float, optional
            First-order decay rate [1/T], by default 0
        retardation_factor : float, optional
            Retardation factor for tracer (dimensionless), by default 1.0
        evapoconcentration_factor : float, optional
            Scaling factor for evapoconcentration, by default 1.0
        dt : float, optional
            Time step of simulation, by default 1.0
        run_NMBC_Q : bool, optional
            Boolean defining if analytical mass breakthrough curves for discharge should be computed
        run_NMBC_ET : bool, optional
            Boolean defining if analytical mass breakthrough curves for evapotranspiration should be computed
        run_NMBC_react : bool, optional
            Boolean defining if analytical mass breakthrough curves for reacted mass should be computed
        run_water_backward_TTD : bool, optional
            Boolean defining if analytical water backward TTD should be computed
        run_tracer_backward_TTD : bool, optional
            Boolean defining if analytical tracer backward TTD should be computed
        max_age_tracked : float, optional
            Maximum age tracked in NMBC and backwards TTDs.
            If no input, will be equal to the timeseries length.
        """
        self.initial_storage = initial_storage
        self.initial_mass = initial_mass
        self.reaction_rate = reaction_rate
        self.retardation_factor = retardation_factor
        self.evapoconcentration_factor = evapoconcentration_factor
        self.dt = dt
        self.run_NMBC_Q = run_NMBC_Q
        self.run_NMBC_ET = run_NMBC_ET
        self.run_NMBC_react = run_NMBC_react
        self.run_water_bTTD = run_water_bTTD
        self.run_tracer_bTTD = run_tracer_bTTD
        self.max_age_tracked = max_age_tracked

    def set_input(self, input_tuple, output_tuple):
        """
        Sets the input and output fluxes.

        Parameters
        ----------
        input_tuple : list(Tuple)
            [(water_flux, mass_flux)], each a numpy array [length T]
        output_tuple : list(Tuple)
            [(Q (discharge), ET (evapotranspiration))], each a numpy array [length T]
        """
        self.input_water_flux = input_tuple[0]
        self.input_mass_flux = input_tuple[1]

        self.output_Q = output_tuple[0]
        self.output_ET = output_tuple[1]

    def solve_storage(self):
        """
        Solves water storage using simple water balance.
        """
        self.storage = np.zeros(len(self.input_water_flux) + 1)
        self.storage[0] = self.initial_storage

        net_flux = self.input_water_flux - self.output_Q - self.output_ET
        self.storage[1:] = self.initial_storage + np.cumsum(net_flux * self.dt)

    def solve_storage_mass(self):
        """
        Solves tracer mass in storage using Runge-Kutta 4th-order integration.
        The output mass fluxes are estimated a posteriori from RK4-averaged masses/concentrations.

        ODE:
        dMs/dt = m_in - Q*(1/R)*C - ET*alpha*(1/R)*C - k*Ms
        Where C = Ms / S
        """
        eps = 1e-10  # For numerical stability

        self.storage_mass = np.zeros(len(self.input_water_flux) + 1)
        self.storage_mass[0] = self.initial_mass

        self.output_Q_mass = np.zeros_like(self.input_mass_flux, dtype=float)
        self.output_ET_mass = np.zeros_like(self.input_mass_flux, dtype=float)
        self.output_reacted_mass = np.zeros_like(self.input_mass_flux, dtype=float)

        def rhs(M_s, m_in, J, Q, ET, S, intermediate_dt):
            S_eff = S + (J - Q - ET) * intermediate_dt
            S_eff = max(S_eff, eps)
            C = M_s / S_eff
            return (
                m_in
                - Q * C / self.retardation_factor
                - ET * C * self.evapoconcentration_factor / self.retardation_factor
                - self.reaction_rate * M_s
            )

        for i, (J, m_in, Q, ET) in enumerate(zip(
                self.input_water_flux, self.input_mass_flux,
                self.output_Q, self.output_ET)):

            M = self.storage_mass[i]
            S = self.storage[i]

            # RK4 steps
            k1 = rhs(M, m_in, J, Q, ET, S, intermediate_dt=0)
            k2 = rhs(M + 0.5 * self.dt * k1, m_in, J, Q, ET, S, intermediate_dt=0.5 * self.dt)
            k3 = rhs(M + 0.5 * self.dt * k2, m_in, J, Q, ET, S, intermediate_dt=0.5 * self.dt)
            k4 = rhs(M + self.dt * k3, m_in, J, Q, ET, S, intermediate_dt=self.dt)

            M_next = M + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self.storage_mass[i + 1] = M_next

            # Compute mass decay using RK4 averaging (based on average rate of change of the state variable)
            self.output_reacted_mass[i] = (
                self.reaction_rate
                * (M + 2 * (M + 0.5 * k1 * self.dt) + 2 * (M + 0.5 * k2 * self.dt) + (M + k3 * self.dt)) / 6
            )

            # Compute streamflow and evaporation tracer flux from mass balance
            Cs_bar = (
                (M - M_next + self.dt * (m_in - self.output_reacted_mass[i]))
                / (Q + self.evapoconcentration_factor * ET)
            )

            self.output_Q_mass[i] = Q * Cs_bar
            self.output_ET_mass[i] = ET * self.evapoconcentration_factor * Cs_bar

    def solve_NMBC_Q(self):
        """
        Solves the analytical solution for the normalized mass breakthrough
        curve in discharge.
        """
        input_timeseries_len = len(self.input_mass_flux)

        if self.max_age_tracked is None:
            num_age_steps = input_timeseries_len
        else:
            num_age_steps = self.max_age_tracked

        NMBC = np.zeros((np.count_nonzero(self.input_mass_flux), num_age_steps), dtype=float)
        idx_mass_inputs = np.nonzero(self.input_mass_flux)[0]

        for j in range(len(NMBC)):
            t = idx_mass_inputs[j]
            S_bar = (
                (5 * self.storage[t:-1]
                 + 2 * self.dt * (self.input_water_flux[t:] - self.output_Q[t:] - self.output_ET[t:])
                 + self.storage[t+1:])
                / 6
            )

            norm_out = (
                ((self.output_Q[t:] + self.evapoconcentration_factor * self.output_ET[t:])
                 / (self.retardation_factor * S_bar)
                 + self.reaction_rate)
                * self.dt
            )

            if input_timeseries_len-t > num_age_steps:
                NMBC[j, :] = (
                    self.output_Q[t:t+num_age_steps] / (self.retardation_factor * S_bar[:num_age_steps])
                    * np.exp(-np.cumsum(norm_out[:num_age_steps]))
                    )
            else:
                NMBC[j, :input_timeseries_len-t] = (
                    self.output_Q[t:] / (self.retardation_factor * S_bar)
                    * np.exp(-np.cumsum(norm_out))
                    )

        self.NMBC_Q = NMBC

    def solve_NMBC_ET(self):
        """
        Solves the analytical solution for the normalized mass breakthrough
        curve in evapotranspiration.
        """
        input_timeseries_len = len(self.input_mass_flux)

        if self.max_age_tracked is None:
            num_age_steps = input_timeseries_len
        else:
            num_age_steps = self.max_age_tracked

        NMBC = np.zeros((np.count_nonzero(self.input_mass_flux), num_age_steps), dtype=float)
        idx_mass_inputs = np.nonzero(self.input_mass_flux)[0]

        for j in range(len(NMBC)):
            t = idx_mass_inputs[j]
            S_bar = (
                (5 * self.storage[t:-1]
                 + 2 * self.dt * (self.input_water_flux[t:] - self.output_Q[t:] - self.output_ET[t:])
                 + self.storage[t+1:])
                / 6
            )

            norm_out = (
                ((self.output_Q[t:] + self.evapoconcentration_factor * self.output_ET[t:])
                 / (self.retardation_factor * S_bar)
                 + self.reaction_rate)
                * self.dt
            )

            if input_timeseries_len-t > num_age_steps:
                NMBC[j, :] = (
                    self.evapoconcentration_factor * self.output_ET[t:t+num_age_steps]
                    / (self.retardation_factor * S_bar[:num_age_steps])
                    * np.exp(-np.cumsum(norm_out[:num_age_steps]))
                    )
            else:
                NMBC[j, :input_timeseries_len-t] = (
                    self.evapoconcentration_factor * self.output_ET[t:]
                    / (self.retardation_factor * S_bar)
                    * np.exp(-np.cumsum(norm_out))
                    )

        self.NMBC_ET = NMBC

    def solve_NMBC_react(self):
        """
        Solves the analytical solution for the normalized mass breakthrough
        curve in reacted mass.
        """
        input_timeseries_len = len(self.input_mass_flux)

        if self.max_age_tracked is None:
            num_age_steps = input_timeseries_len
        else:
            num_age_steps = self.max_age_tracked

        NMBC = np.zeros((np.count_nonzero(self.input_mass_flux), num_age_steps), dtype=float)
        idx_mass_inputs = np.nonzero(self.input_mass_flux)[0]

        for j in range(len(NMBC)):
            t = idx_mass_inputs[j]
            S_bar = (
                (5 * self.storage[t:-1]
                 + 2 * self.dt * (self.input_water_flux[t:] - self.output_Q[t:] - self.output_ET[t:])
                 + self.storage[t+1:])
                / 6
            )

            norm_out = (
                ((self.output_Q[t:] + self.evapoconcentration_factor * self.output_ET[t:])
                 / (self.retardation_factor * S_bar)
                 + self.reaction_rate)
                * self.dt
            )

            if input_timeseries_len-t > num_age_steps:
                NMBC[j, :] = (
                    self.reaction_rate * np.exp(-np.cumsum(norm_out[:num_age_steps]))
                    )
            else:
                NMBC[j, :input_timeseries_len-t] = (
                    self.reaction_rate * np.exp(-np.cumsum(norm_out))
                    )

        self.NMBC_react = NMBC

    def solve_water_backward_TTD(self):
        """
        Solves the analytical solution for the backwards tracer TTD in storage.
        """
        output_timeseries_len = len(self.output_Q)

        if self.max_age_tracked is None:
            num_age_steps = output_timeseries_len
        else:
            num_age_steps = self.max_age_tracked

        water_backward_TTD = np.zeros([output_timeseries_len, num_age_steps])

        reversed_input_water_flux = self.input_water_flux[::-1]
        S_bar = (self.storage[:-1] + self.storage[1:]) / 2

        for t in range(output_timeseries_len):
            S_bar_array = (
                (5 * self.storage[:t]
                 + 2 * self.dt * (self.input_water_flux[:t] - self.output_Q[:t] - self.output_ET[:t])
                 + self.storage[1:t+1])
                / 6
            )

            norm_out = (
                ((self.output_Q[:t] + self.output_ET[:t]) / S_bar_array)
                * self.dt
            )

            if t > 0:
                water_TTD_temp = (
                        reversed_input_water_flux[output_timeseries_len-t:] / S_bar[t]
                        * np.exp(-np.cumsum(norm_out[t-1::-1]))
                        )

                if t < num_age_steps:
                    water_backward_TTD[t, :t] = water_TTD_temp
                else:
                    water_backward_TTD[t, :] = water_TTD_temp[:num_age_steps]

        self.water_bTTD = water_backward_TTD

    def solve_tracer_backward_TTD(self):
        """
        Solves the analytical solution for the backwards tracer TTD in storage.
        """
        output_timeseries_len = len(self.output_Q_mass)

        if self.max_age_tracked is None:
            num_age_steps = output_timeseries_len
        else:
            num_age_steps = self.max_age_tracked

        tracer_backward_TTD = np.zeros([output_timeseries_len, num_age_steps])

        reversed_input_mass_flux = self.input_mass_flux[::-1]
        M_bar = (self.storage_mass[:-1] + self.storage_mass[1:]) / 2

        for t in range(output_timeseries_len):
            S_bar_array = (
                (5 * self.storage[:t]
                 + 2 * self.dt * (self.input_water_flux[:t] - self.output_Q[:t] - self.output_ET[:t])
                 + self.storage[1:t+1])
                / 6
            )

            norm_out = (
                ((self.output_Q[:t] + self.evapoconcentration_factor * self.output_ET[:t])
                 / (self.retardation_factor * S_bar_array)
                 + self.reaction_rate)
                * self.dt
            )

            tracer_TTD_temp = (
                    reversed_input_mass_flux[output_timeseries_len-t:] / M_bar[t]
                    * np.exp(-np.cumsum(norm_out[t-1::-1]))
                    )

            if t < num_age_steps:
                tracer_backward_TTD[t, :t] = tracer_TTD_temp
            else:
                tracer_backward_TTD[t, :] = tracer_TTD_temp[:num_age_steps]

        self.tracer_bTTD = tracer_backward_TTD

    def run(self):
        """
        Executes the main simulation routine by solving storage and mass balance,
        and conditionally running additional modules based on specified flags.
        """

        self.solve_storage()
        self.solve_storage_mass()

        if self.run_NMBC_Q is True:
            self.solve_NMBC_Q()

        if self.run_NMBC_ET is True:
            self.solve_NMBC_ET()

        if self.run_NMBC_react is True:
            self.solve_NMBC_react()

        if self.run_tracer_bTTD is True:
            self.solve_tracer_backward_TTD()

        if self.run_water_bTTD is True:
            self.solve_water_backward_TTD()
