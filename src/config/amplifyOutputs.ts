type AmplifyOutputs = {
  custom: {
    ENV: {
      amplify_env: string;
    };
    API: Record<string, { endpoint?: string }>;
    WSS: Record<string, { endpoint?: string }>;
  };
  storage: {
    bucket_name: string;
  };
};

const matches = import.meta.glob("/amplify_outputs.json", { eager: true }) as Record<
  string,
  { default?: AmplifyOutputs }
>;

const rootOutputs = matches["/amplify_outputs.json"]?.default;

const fallbackOutputs: AmplifyOutputs = {
  custom: {
    ENV: { amplify_env: "local" },
    API: {
      GenASLAPIlocal: { endpoint: "API_NOT_CONFIGURED" },
      GenASLAPImain: { endpoint: "API_NOT_CONFIGURED" },
    },
    WSS: {
      GenASLWSSlocal: { endpoint: "WSS_NOT_CONFIGURED" },
      GenASLWSSmain: { endpoint: "WSS_NOT_CONFIGURED" },
    },
  },
  storage: {
    bucket_name: "BUCKET_NOT_CONFIGURED",
  },
};

export default rootOutputs ?? fallbackOutputs;
