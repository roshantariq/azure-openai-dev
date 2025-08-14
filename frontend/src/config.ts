// Type-safe loader for CRA env vars (available at build time)
const required = (key: string) => {
  const v = process.env[key];
  if (!v) throw new Error(`Missing required env var: ${key}`);
  return v;
};

export const POWERBI_CONFIG = {
  workspaceId: required('REACT_APP_POWERBI_WORKSPACE_ID'),
  reportId:    required('REACT_APP_POWERBI_REPORT_ID'),
  defaultPage: process.env.REACT_APP_POWERBI_DEFAULT_PAGE || undefined,
};
