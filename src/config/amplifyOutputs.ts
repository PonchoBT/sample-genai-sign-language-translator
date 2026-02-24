type AnyRecord = Record<string, any>;

const outputModules = import.meta.glob("../../amplify_outputs.json", {
  eager: true,
}) as Record<string, { default: AnyRecord }>;

export const amplifyOutputs: AnyRecord | null =
  Object.values(outputModules)[0]?.default ?? null;

export const amplifyEnv: string =
  amplifyOutputs?.custom?.ENV?.amplify_env ?? "main";

export const restApiEndpoint: string =
  amplifyOutputs?.custom?.API?.[`GenASLAPI${amplifyEnv}`]?.endpoint ??
  amplifyOutputs?.custom?.API?.GenASLAPImain?.endpoint ??
  "API_NOT_CONFIGURED";

export const wssApiEndpoint: string =
  amplifyOutputs?.custom?.WSS?.[`GenASLWSS${amplifyEnv}`]?.endpoint ??
  amplifyOutputs?.custom?.WSS?.GenASLWSSmain?.endpoint ??
  "WSS_NOT_CONFIGURED";

export const storageBucketName: string =
  amplifyOutputs?.storage?.bucket_name ?? "";

export const hasAuthUserPoolConfig: boolean = Boolean(
  amplifyOutputs?.auth?.user_pool_id &&
  amplifyOutputs?.auth?.user_pool_client_id
);
