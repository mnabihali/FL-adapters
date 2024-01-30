import flwr as fl


class STRATEGY(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        state_dict = torch.load("./pretrained_models/librispeech1000.pth")
        net.load_state_dict(state_dict)
        torch.cuda.empty_cache()
        gc.collect()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size = random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        # print("CONFIG.  sample:", sample_size,"min_num_clients:", min_num_clients, "n_clients:",n_clients)
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[BaseException],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if self.accept_failures and failures:
            return None, {}

        key_name = "train_loss" if weight_strategy == "loss" else "wer"

        weights = None

        if weight_strategy == 'num':
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            weights = aggregate(weights_results)

        elif weight_strategy == "loss" or weight_strategy == "wer":
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results
            ]
            weights = aggregate(weights_results)

        if weights is not None:

            params_dict = zip(net.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"trained_models/{_MODELNAME}-round-{server_round}.pth")
            new_parameters = get_parameters(net)
            return ndarrays_to_parameters(new_parameters), {}
        else:
            print(f"returning None weights, something went wrongh during aggregation..... !!!!!!!!!!!!!!!")
            return ndarrays_to_parameters(weights), {}
