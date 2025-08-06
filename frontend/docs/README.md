# Overview of Components

Here is the hierarchy of components used by the WISE frontend.

```
App
|- WiseHeader
|  |- SearchDropdown
|  |- ...
|- SearchResults
|  |- StillImageView
|  |- ImageDetailsModal
|  |- ...
|
.
```

* App : The top level component for WISE user interface which is defined in [App.tsx](../src/App.tsx). This is the parent component for all other components in the frontend.

* WiseHeader : This components is defined in [WiseHeaders.tsx](../src/WiseHeader.tsx) and is responsible for managing the following three items: (a) the dropdown for selecting the search target (or modality like Video, Face, Object, Audio, Metadata, etc.), (b) the search input bar, (c) the pop up panel that activates when the search input is focused. This component uses the [`WiseHeaderProps`](../src/misc/types.ts) passed to it by App (i.e. its parent component).
  - SearchDropdown : This components is defined in [WiseHeaders.tsx](../WiseHeaders.tsx) and is responsible for managing the panel that pops up when the search input bar (defined by WiseHeader) is focused or activated.
  - ...
