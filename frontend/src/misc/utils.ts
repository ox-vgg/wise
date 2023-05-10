// Adapted from https://stackoverflow.com/a/57888548
export const fetchWithTimeout = (url: string, ms: number, { signal, ...options }: { signal?: AbortSignal } | RequestInit = {}) => {
  const controller = new AbortController();
  const promise = fetch(url, { signal: controller.signal, ...options });
  if (signal) signal.addEventListener("abort", () => controller.abort());
  const timeout = setTimeout(() => controller.abort("Request timed out"), ms);
  return promise.catch((err) => {
    console.log('CAGHT', controller.signal)
    if (controller.signal.aborted) {
      throw new Error('Request timed out', { cause: err });
      // TODO use controller.signal.reason (issue: the reason parameter is not supported by some browsers)
    } else {
      throw err
    }
  }).finally(() => clearTimeout(timeout));
};

export const chunk = (arr: any[], chunkSize: number) => {
  let R = [];
  for (let i = 0, len = arr.length; i < len; i += chunkSize)
    R.push(arr.slice(i, i + chunkSize));
  return R;
}

export const getArrayOfEmptyArrays = (length: number): any[][] => {
  return Array.from(Array(Math.ceil(length))).map(_ => [])
}