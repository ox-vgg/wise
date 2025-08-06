import { ProjectInfo, ViewModality } from "./types";

// Adapted from https://stackoverflow.com/a/57888548
export const fetchWithTimeout = (url: string, ms: number, { signal, ...options }: { signal?: AbortSignal } | RequestInit = {}) => {
  const controller = new AbortController();
  const promise = fetch(url, { signal: controller.signal, ...options });
  if (signal) signal.addEventListener("abort", () => controller.abort());
  const timeout = setTimeout(() => controller.abort("Request timed out"), ms);
  return promise.catch((err) => {
    console.log('CAUGHT', controller.signal)
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

export const interleaveArrayWithElement = <T>(elems: T[], divider: T): T[] => {
  const divided_elems = [];
  for (let i = 0; i < ((elems.length *2) -1); i++) {
    if ((i % 2) == 0)
      divided_elems.push(elems[i/2]);
    else
      divided_elems.push(divider);
    }
  return divided_elems;
}


// This incantation removes an object property that may not exist
// without triggering TypeScript TS2339.
export const excludeKey = <T extends object, U extends keyof any>(obj: T, key: U) => {
  const { [key]: _, ...newObj } = obj;
  return newObj;
}
export const secondsToMinSecPadded = (time: number) => {
  const minutes = Math.floor(time / 60);
  const seconds = `${Math.floor(time % 60)}`.padStart(2, "0");
  return `${minutes}:${seconds}`;
};

const _clamp = (min: number, max: number) => {
    return (x: number) => Math.min(max, Math.max(min, x));
}
export const clamp_bbox = _clamp(0, 1);

type NonNullSearchTargets = NonNullable<ProjectInfo['search_targets']>;
type SearchTargets = keyof NonNullSearchTargets;
export const is_metadata_supported = (projectInfo: ProjectInfo): boolean => {
  const search_targets = projectInfo.search_targets;
  if (!search_targets || Object.keys(search_targets).length === 0) return false;
  return Object.keys(search_targets).some((media_type) => {
    const _key = media_type as SearchTargets;
    const _targets = search_targets[_key] as NonNullable<NonNullSearchTargets[SearchTargets]>;
    return _targets.includes('wise/metadata');
  });
}

export const is_metadata_filter_supported = (projectInfo: ProjectInfo, viewModality: ViewModality): boolean => {
  const search_targets = projectInfo.search_targets;
  if (!search_targets || Object.keys(search_targets).length === 0) return false;
  const media_type = (viewModality === 'VideoAudio' ? 'audio' : viewModality.toLowerCase() as SearchTargets);
  return search_targets[media_type]?.includes('wise/metadata') ?? false;
}