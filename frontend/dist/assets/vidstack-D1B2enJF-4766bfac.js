import{m as p,n as h,R as E,B as l,C as k,D as g,b as c,l as f,e as y,E as C,L as A}from"./index-ab6fc9a5.js";import{a as v,b as L,g as b,h as m,l as w,c as S}from"./vidstack-BNZ1p4l6-9c386d8b.js";import"../config.js";class j{constructor(t){this.ab=new chrome.cast.media.MediaInfo(t.src,t.type)}build(){return this.ab}mj(t){return t.includes("live")?this.ab.streamType=chrome.cast.media.StreamType.LIVE:this.ab.streamType=chrome.cast.media.StreamType.BUFFERED,this}nj(t){return this.ab.tracks=t.map(this.oj),this}pj(t,e){return this.ab.metadata=new chrome.cast.media.GenericMediaMetadata,this.ab.metadata.title=t,this.ab.metadata.images=[{url:e}],this}oj(t,e){const s=new chrome.cast.media.Track(e,chrome.cast.media.TrackType.TEXT);return s.name=t.label,s.trackContentId=t.src,s.trackContentType="text/vtt",s.language=t.language,s.subtype=t.kind.toUpperCase(),s}}const d=chrome.cast.media.TrackType.TEXT,T=chrome.cast.media.TrackType.AUDIO;class D{constructor(t,e,s){this.pd=t,this.b=e,this.Be=s}ie(){const t=this.vg.bind(this);f(this.b.audioTracks,"change",t),f(this.b.textTracks,"mode-change",t),y(this.qj.bind(this))}od(){return this.b.$state.textTracks().filter(t=>t.src&&t.type==="vtt")}wg(){return this.b.$state.audioTracks()}yc(t){var s;const e=((s=this.pd.mediaInfo)==null?void 0:s.tracks)??[];return t?e.filter(i=>i.type===t):e}rj(){const t=[],e=this.wg().find(i=>i.selected),s=this.od().filter(i=>i.mode==="showing");if(e){const i=this.yc(T),r=this.Ae(i,e);r&&t.push(r.trackId)}if(s!=null&&s.length){const i=this.yc(d);if(i.length)for(const r of s){const a=this.Ae(i,r);a&&t.push(a.trackId)}}return t}qj(){const t=this.od();if(!this.pd.isMediaLoaded)return;const e=this.yc(d);for(const s of t)if(!this.Ae(e,s)){C(()=>{var r;return(r=this.Be)==null?void 0:r.call(this)});break}}sj(t){if(!this.pd.isMediaLoaded)return;const e=this.wg(),s=this.od(),i=this.yc(T),r=this.yc(d);for(const a of i){if(this.xg(e,a))continue;const n={id:a.trackId.toString(),label:a.name,language:a.language,kind:a.subtype??"main",selected:!1};this.b.audioTracks[A.ea](n,t)}for(const a of r){if(this.xg(s,a))continue;const n={id:a.trackId.toString(),src:a.trackContentId,label:a.name,language:a.language,kind:a.subtype.toLowerCase()};this.b.textTracks.add(n,t)}}vg(t){if(!this.pd.isMediaLoaded)return;const e=this.rj(),s=new chrome.cast.media.EditTracksInfoRequest(e);this.tj(s).catch(i=>{})}tj(t){const e=b();return new Promise((s,i)=>e==null?void 0:e.editTracksInfo(t,s,i))}xg(t,e){return t.find(s=>this.yg(s,e))}Ae(t,e){return t.find(s=>this.yg(e,s))}yg(t,e){return e.name===t.label&&e.language===t.language&&e.subtype.toLowerCase()===t.kind.toLowerCase()}}class R{constructor(t,e){this.f=t,this.b=e,this.$$PROVIDER_TYPE="GOOGLE_CAST",this.scope=p(),this.L=null,this.Aa="disconnected",this.va=0,this.ha=0,this.ca=new h(0,0),this.Ba=new h(0,0),this.ga=new E(this.lc.bind(this)),this.Qa=null,this.Ce=!1,this.wa=new D(this.f,this.b,this.Be.bind(this))}get c(){return this.b.delegate.c}get type(){return"google-cast"}get currentSrc(){return this.L}get player(){return this.f}get cast(){return v()}get session(){return L()}get media(){return b()}get hasActiveSession(){return m(this.L)}setup(){this.uj(),this.vj(),this.wa.ie(),this.c("provider-setup",this)}uj(){w(cast.framework.CastContextEventType.CAST_STATE_CHANGED,this.Ag.bind(this))}vj(){const t=cast.framework.RemotePlayerEventType,e={[t.IS_CONNECTED_CHANGED]:this.Ag,[t.IS_MEDIA_LOADED_CHANGED]:this.Bg,[t.CAN_CONTROL_VOLUME_CHANGED]:this.Cg,[t.CAN_SEEK_CHANGED]:this.Dg,[t.DURATION_CHANGED]:this.ee,[t.IS_MUTED_CHANGED]:this.Oa,[t.VOLUME_LEVEL_CHANGED]:this.Oa,[t.IS_PAUSED_CHANGED]:this.wj,[t.LIVE_SEEKABLE_RANGE_CHANGED]:this.ob,[t.PLAYER_STATE_CHANGED]:this.xj};this.zg=e;const s=this.yj.bind(this);for(const i of l(e))this.f.controller.addEventListener(i,s);k(()=>{for(const i of l(e))this.f.controller.removeEventListener(i,s)})}async play(){var t;if(!(!this.f.isPaused&&!this.Ce)){if(this.Ce){await this.Eg(!1,0);return}(t=this.f.controller)==null||t.playOrPause()}}async pause(){var t;this.f.isPaused||(t=this.f.controller)==null||t.playOrPause()}getMediaStatus(t){return new Promise((e,s)=>{var i;(i=this.media)==null||i.getStatus(t,e,s)})}setMuted(t){var s;(t&&!this.f.isMuted||!t&&this.f.isMuted)&&((s=this.f.controller)==null||s.muteOrUnmute())}setCurrentTime(t){var e;this.f.currentTime=t,this.c("seeking",t),(e=this.f.controller)==null||e.seek()}setVolume(t){var e;this.f.volumeLevel=t,(e=this.f.controller)==null||e.setVolumeLevel()}async loadSource(t){var i;if(((i=this.Qa)==null?void 0:i.src)!==t&&(this.Qa=null),m(t)){this.zj(),this.L=t;return}this.c("load-start");const e=this.Aj(t),s=await this.session.loadMedia(e);if(s){this.L=null,this.c("error",Error(S(s)));return}this.L=t}destroy(){this.A(),this.Fg()}A(){this.Qa||(this.ha=0,this.ca=new h(0,0),this.Ba=new h(0,0)),this.ga.aa(),this.va=0,this.Qa=null}zj(){const t=new g("resume-session",{detail:this.session});this.Bg(t);const{muted:e,volume:s,savedState:i}=this.b.$state,r=i();this.setCurrentTime(Math.max(this.f.currentTime,(r==null?void 0:r.currentTime)??0)),this.setMuted(e()),this.setVolume(s()),(r==null?void 0:r.paused)===!1&&this.play()}Fg(){this.cast.endCurrentSession(!0);const{remotePlaybackLoader:t}=this.b.$state;t.set(null)}Bj(){const{savedState:t}=this.b.$state;t.set({paused:this.f.isPaused,currentTime:this.f.currentTime}),this.Fg()}lc(){this.Cj()}yj(t){this.zg[t.type].call(this,t)}Ag(t){const e=this.cast.getCastState(),s=e===cast.framework.CastState.CONNECTED?"connected":e===cast.framework.CastState.CONNECTING?"connecting":"disconnected";if(this.Aa===s)return;const i={type:"google-cast",state:s},r=this.bb(t);this.Aa=s,this.c("remote-playback-change",i,r),s==="disconnected"&&this.Bj()}Bg(t){if(!!!this.f.isMediaLoaded)return;const s=c(this.b.$state.source);Promise.resolve().then(()=>{if(s!==c(this.b.$state.source)||!this.f.isMediaLoaded)return;this.A();const i=this.f.duration;this.Ba=new h(0,i);const r={provider:this,duration:i,buffered:this.ca,seekable:this.Gg()},a=this.bb(t);this.c("loaded-metadata",void 0,a),this.c("loaded-data",void 0,a),this.c("can-play",r,a),this.Cg(),this.Dg(t);const{volume:o,muted:n}=this.b.$state;this.setVolume(o()),this.setMuted(n()),this.ga.Ya(),this.wa.sj(a),this.wa.vg(a)})}Cg(){this.b.$state.canSetVolume.set(this.f.canControlVolume)}Dg(t){const e=this.bb(t);this.c("stream-type-change",this.Dj(),e)}Dj(){var e;return((e=this.f.mediaInfo)==null?void 0:e.streamType)===chrome.cast.media.StreamType.LIVE?this.f.canSeek?"live:dvr":"live":"on-demand"}Cj(){if(this.Qa)return;const t=this.f.currentTime;if(t===this.va)return;const e=this.ha,s=this.vc(t),i={currentTime:t,played:s};this.c("time-update",i),t>e&&this.ob(),this.b.$state.seeking()&&this.c("seeked",t),this.va=t}vc(t){return this.ha>=t?this.ca:this.ca=new h(0,this.ha=t)}ee(t){if(!this.f.isMediaLoaded||this.Qa)return;const e=this.f.duration,s=this.bb(t);this.Ba=new h(0,e),this.c("duration-change",e,s)}Oa(t){if(!this.f.isMediaLoaded)return;const e={muted:this.f.isMuted,volume:this.f.volumeLevel},s=this.bb(t);this.c("volume-change",e,s)}wj(t){const e=this.bb(t);this.f.isPaused?this.c("pause",void 0,e):this.c("play",void 0,e)}ob(t){const e={seekable:this.Gg(),buffered:this.ca},s=t?this.bb(t):void 0;this.c("progress",e,s)}xj(t){const e=this.f.playerState,s=chrome.cast.media.PlayerState;if(this.Ce=e===s.IDLE,e===s.PAUSED)return;const i=this.bb(t);switch(e){case s.PLAYING:this.c("playing",void 0,i);break;case s.BUFFERING:this.c("waiting",void 0,i);break;case s.IDLE:this.ga.aa(),this.c("pause"),this.c("end");break}}Gg(){return this.f.liveSeekableRange?new h(this.f.liveSeekableRange.start,this.f.liveSeekableRange.end):this.Ba}bb(t){return t instanceof Event?t:new g(t.type,{detail:t})}Ej(t){const{streamType:e,title:s,poster:i}=this.b.$state;return new j(t).pj(s(),i()).mj(e()).nj(this.wa.od()).build()}Aj(t){var r,a;const e=this.Ej(t),s=new chrome.cast.media.LoadRequest(e),i=this.b.$state.savedState();return s.autoplay=(((r=this.Qa)==null?void 0:r.paused)??(i==null?void 0:i.paused))===!1,s.currentTime=((a=this.Qa)==null?void 0:a.time)??(i==null?void 0:i.currentTime)??0,s}async Eg(t,e){const s=c(this.b.$state.source);this.Qa={src:s,paused:t,time:e},await this.loadSource(s)}Be(){this.Eg(this.f.isPaused,this.f.currentTime).catch(t=>{})}}export{R as GoogleCastProvider};
