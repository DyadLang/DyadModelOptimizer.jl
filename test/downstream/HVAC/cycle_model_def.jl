using HVAC, ModelingToolkit, SciMLBase,
      ModelingToolkitStandardLibrary, ModelingToolkitStandardLibrary.Blocks
import HVAC: moistair

function build_model(; patch_params = true)
    @debug "Testing a *complete* cycle model with moist air"
    # 1. Load the refrigerant:
    ref_name = "R32"
    refrigerant = load_refrigerant(ref_name * ".yaml")

    # 2. Choose initial conditions
    # 2.a. Refrigerant side: Evaporating and Condensing pressure
    #P_cond_in, P_cond_out, P_evap_in, P_evap_out = 2.6e6, 2.55e6, 1.0e6, 0.9e6
    P_cond_in, P_cond_out, P_evap_in, P_evap_out = 1.6e6, 1.5e6, 1.2e6, 1.1e6

    # Enthalpy of the refrigerant at the inlet and outlet of the condenser and evaporator
    #h_cond_in, h_cond_out, h_evap_in, h_evap_out = 520e3, 230e3, 230e3, 430e3
    h_cond_in, h_cond_out, h_evap_in, h_evap_out = 320e3, 230e3, 230e3, 330e3

    # Refrigerant mass flow rate:
    ref_mdot_start = 0.0001

    # Evaporating and condensing temperature of the refrigerant
    T_cond, T_evap = 300.15, 285.15

    # 2.b. Air side:
    # Air mass flow rate
    air_mdot_in = 40.0

    air_p_in_start = 101325.0
    air_p_out_start = 101325.0 - 100.0

    # Moist air relative humidities for indoor and outdoor air
    ϕ_cond, ϕ_evap = 0.5, 0.6

    cond_air_T_in_start = HVAC.from_degC(40)
    cond_air_T_out_start = HVAC.from_degC(22)

    evap_air_T_in_start = T_evap + 25.0
    evap_air_T_out_start = HVAC.from_degC(18.0)

    # 3. Choose controller inputs:
    # Control Inputs
    compressor_speed_start = 10
    LEV_position_start = 220

    # %%
    # 4. Components:
    # 4.1. Compressor
    compressor_record = MERLCompressorRecord(;
        P_a_start = P_evap_out, P_b_start = P_cond_in,
        m_flow_start = ref_mdot_start, speed_start = compressor_speed_start,
        hSuction_start = h_evap_out, hDischarge_start = h_cond_in)

    @named compressor = TypicalCompressor(compressor_record)
    @named compressor_speed_signal = Constant(; k = compressor_speed_start)

    # 4.2 Valve:
    LEV_record = MERLLEVRecord(;
        LEV_position_start, P_a_start = P_cond_out, m_flow_start = ref_mdot_start,
        h_in_start = h_cond_out, P_b_start = P_evap_in, h_out_start = h_evap_in)
    @named LEV = TypicalLEV(LEV_record)
    # Valve control input:
    @named LEV_position_signal = Constant(; k = LEV_position_start)

    # Common condenser and evaporator params:
    nTube = 1
    nSeg = 4
    Lt = 30
    Di = 2.54e-2
    Do = Di + 0.5e-3

    # 4.3. Condenser:
    condenser_record = MERLTubeFinHEXRecord(; nTube, nSeg, Lt, Di, Do,
        ref_model_structure = av_vb,
        ref_p_in_start = P_cond_in,
        ref_p_out_start = P_cond_out,
        ref_h_in_start = h_cond_in,
        ref_h_out_start = h_cond_out,
        ref_m_flow_start = ref_mdot_start,
        wall_T_in_start = HVAC.from_degC(27),
        wall_T_out_start = HVAC.from_degC(27),
        air_p_in_start,
        air_p_out_start,
        air_m_air_flow_start = air_mdot_in,
        air_T_in_start = cond_air_T_in_start,
        air_T_out_start = cond_air_T_out_start,
        air_Xi_in_start = moistair.Xi_ref,
        air_Xi_out_start = moistair.Xi_ref,
        air_Q_start = 50
    )

    @named condenser = TubeFinHEX(condenser_record)

    # Condenser Air side:
    @named condAirSource = MassFlowSource_Tϕ(; T_in = cond_air_T_in_start, ϕ_in = ϕ_cond,
        m_flow_in = air_mdot_in, P_start = air_p_in_start)
    @named condAirSink = Boundary_PTϕ(;
        P_in = air_p_out_start, T_in = cond_air_T_out_start,
        ϕ_in = ϕ_cond, m_flow_start = air_mdot_in)

    # Evaporator refrigerant side:
    evaporator_record = MERLTubeFinHEXRecord(; nTube, nSeg, Lt, Di, Do,
        ref_model_structure = av_vb,
        ref_p_in_start = P_evap_in,
        ref_p_out_start = P_evap_out,
        ref_h_in_start = h_evap_in,
        ref_h_out_start = h_evap_out,
        ref_m_flow_start = ref_mdot_start,
        wall_T_in_start = HVAC.from_degC(24),
        wall_T_out_start = HVAC.from_degC(24),
        air_p_in_start,
        air_p_out_start,
        air_m_air_flow_start = air_mdot_in,
        air_T_in_start = evap_air_T_in_start,
        air_T_out_start = evap_air_T_out_start,
        air_Xi_in_start = moistair.Xi_ref,
        air_Xi_out_start = moistair.Xi_ref,
        air_Q_start = -50
    )

    @named evaporator = TubeFinHEX(evaporator_record)

    # Evaporator Air side:
    @named evapAirSource = MassFlowSource_Tϕ(; T_in = evap_air_T_in_start, ϕ_in = ϕ_evap,
        m_flow_in = air_mdot_in, P_start = air_p_in_start)
    @named evapAirSink = Boundary_PTϕ(;
        P_in = air_p_out_start, T_in = evap_air_T_out_start,
        ϕ_in = ϕ_evap, m_flow_start = air_mdot_in)

    eqns = [connect(compressor.port_b, condenser.refPort_a)
            connect(condenser.refPort_b, LEV.port_a)
            connect(LEV.port_b, evaporator.refPort_a)
            connect(evaporator.refPort_b, compressor.port_a)
            connect(condAirSource.port, condenser.airPort_a)
            connect(condenser.airPort_b, condAirSink.port1)
            connect(evapAirSource.port, evaporator.airPort_a)
            connect(evaporator.airPort_b, evapAirSink.port1)

            # Control inputs
            connect(compressor_speed_signal.output, compressor.input_speed)
            connect(LEV_position_signal.output, LEV.input)]

    # Define the ODE system:
    systems = [condenser, LEV, evaporator, compressor, condAirSource, condAirSink,
        evapAirSource, evapAirSink, compressor_speed_signal, LEV_position_signal]

    @named sys = ODESystem(eqns, t; systems)

    sysRed = structural_simplify(sys)
    @time sysRed = complete(structural_simplify(sys))

    if patch_params
        model = tunable2guess(sysRed)
    else
        model = sysRed
    end

    return model
end

function tunable2guess(model)
    guesses = filter(
        m -> ModelingToolkit.isparameter(m[1]) &&
            !ModelingToolkit.istunable(m[1], false),
        defaults(model))

    merge!(ModelingToolkit.guesses(model), guesses)
    model
end
