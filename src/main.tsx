import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { Amplify } from "aws-amplify";
import '@aws-amplify/ui-react/styles.css';
import { Authenticator } from '@aws-amplify/ui-react';
import { amplifyOutputs, hasAuthUserPoolConfig } from "./config/amplifyOutputs";

if (amplifyOutputs) {
  Amplify.configure(amplifyOutputs);
  const existingConfig = Amplify.getConfig();
  Amplify.configure({
    ...existingConfig,
    API: {
      ...existingConfig.API,
      REST: amplifyOutputs.custom?.API ?? {},
    },
  });
} else {
  console.warn(
    "No se encontro amplify_outputs.json. El frontend carga, pero Amplify no esta configurado."
  );
}


ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {hasAuthUserPoolConfig ? (
      <Authenticator>
        <App />
      </Authenticator>
    ) : (
      <App />
    )}
  </React.StrictMode>
);
