(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[800],{14682:function(e,t,n){"use strict";n.d(t,{y:function(){return g}});var r=n(9669),o=n.n(r),i=n(84506),s=n(40203),a=n(35823),c=n.n(a),l=o().create({baseURL:i.url_root+"/api/v1/",headers:{"Access-Control-Allow-Origin":"*"}});l.interceptors.request.use(function(e){return e.headers.Authorization="Bearer "+(0,s.Gg)().user.token,e},function(e){console.log("Interceptor request error: "+e)}),l.interceptors.response.use(function(e){return e},function(e){throw console.log(" AXIOS error: "),console.log(e),401===e.response.status&&(0,s.KY)("","",-1,0,"",!0),403===e.response.status&&console.log("Error: "+e.response.data.status),404===e.response.status&&console.log("Error: "+e.response.data.status),409===e.response.status&&console.log("Error: "+e.response.data.status),422===e.response.status&&(0,s.KY)("","",-1,0,"",!0),e});var u=function(e){return e.data},d=function(e){return l.get(e).then(u)},f=function(e,t,n){return l.post(e,{pin:n},{responseType:"blob"}).then(function(e){t=e.headers["content-disposition"].split('"')[1],c()(e.data,t)})},p=function(e,t){return l.post(e,t).then(u)},h=function(e,t){return l.patch(e,t).then(u)},m=function(e){return l.delete(e).then(u)},g={login:function(e){return p("login",e)},getUsers:function(){return d("users")},getUser:function(e){return d("users/".concat(e))},getUserStartupKit:function(e,t,n){return f("users/".concat(e,"/blob"),t,n)},getClientStartupKit:function(e,t,n){return f("clients/".concat(e,"/blob"),t,n)},getOverseerStartupKit:function(e,t){return f("overseer/blob",e,t)},getServerStartupKit:function(e,t,n){return f("servers/".concat(e,"/blob"),t,n)},getClients:function(){return d("clients")},getProject:function(){return d("project")},postUser:function(e){return p("users",e)},patchUser:function(e,t){return h("users/".concat(e),t)},deleteUser:function(e){return m("users/".concat(e))},postClient:function(e){return p("clients",e)},patchClient:function(e,t){return h("clients/".concat(e),t)},deleteClient:function(e){return m("clients/".concat(e))},patchProject:function(e){return h("project",e)},getServer:function(){return d("server")},patchServer:function(e){return h("server",e)}}},12976:function(e,t,n){"use strict";n.d(t,{Z:function(){return C}});var r=n(50029),o=n(64687),i=n.n(o),s=n(41664),a=n.n(s),c=n(29224),l=n.n(c),u=n(70491),d=n.n(u),f=n(86188),p=n.n(f),h=n(85444),m=h.default.div.withConfig({displayName:"styles__StyledLayout",componentId:"sc-xczy9u-0"})(["overflow:hidden;height:100%;width:100%;margin:0;padding:0;display:flex;flex-wrap:wrap;.menu{height:auto;}.content-header{flex:0 0 80px;}"]),g=h.default.div.withConfig({displayName:"styles__StyledContent",componentId:"sc-xczy9u-1"})(["display:flex;flex-direction:column;flex:1 1 0%;overflow:auto;height:calc(100% - 3rem);.inlineeditlarger{padding:10px;}.inlineedit{padding:10px;margin:-10px;}.content-wrapper{padding:",";min-height:800px;}"],function(e){return e.theme.spacing.four}),j=n(11163),b=n(84506),v=n(67294),x=n(40203),y=n(13258),w=n.n(y),P=n(5801),S=n.n(P),_=n(14682),O=n(85893),C=function(e){var t,n=e.children,o=e.headerChildren,s=e.title,c=(0,j.useRouter)(),u=c.pathname,h=c.push,P=(0,v.useState)(),C=P[0],k=P[1],D=(0,x.Gg)();(0,v.useEffect)(function(){_.y.getProject().then(function(e){k(e.project)})},[]);var N=(t=(0,r.Z)(i().mark(function e(){return i().wrap(function(e){for(;;)switch(e.prev=e.next){case 0:(0,x.KY)("none","",-1,0),h("/");case 2:case"end":return e.stop()}},e)})),function(){return t.apply(this,arguments)});return(0,O.jsxs)(m,{children:[(0,O.jsx)(l(),{app:null==C?void 0:C.short_name,appBarActions:D.user.role>0?(0,O.jsxs)("div",{style:{display:"flex",flexDirection:"row",alignItems:"center",marginRight:10},children:[b.demo&&(0,O.jsx)("div",{children:"DEMO MODE"}),(0,O.jsx)(w(),{parentElement:(0,O.jsx)(S(),{icon:{name:"AccountCircleGenericUser",color:"white",size:22},shape:"circle",variant:"link",className:"logout-link"}),position:"top-right",children:(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(y.ActionMenuItem,{label:"Logout",onClick:N}),!1]})})]}):(0,O.jsx)(O.Fragment,{})}),(0,O.jsxs)(p(),{className:"menu",itemMatchPattern:function(e){return e===u},itemRenderer:function(e){var t=e.title,n=e.href;return(0,O.jsx)(a(),{href:n,children:t})},location:u,children:[0==D.user.role&&(0,O.jsxs)(f.MenuContent,{children:[(0,O.jsx)(f.MenuItem,{href:"/",icon:{name:"AccountUser"},title:"Login"}),(0,O.jsx)(f.MenuItem,{href:"/registration-form",icon:{name:"ObjectsClipboardEdit"},title:"User Registration Form"})]}),4==D.user.role&&(0,O.jsxs)(f.MenuContent,{children:[(0,O.jsx)(f.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home"}),(0,O.jsx)(f.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,O.jsx)(f.MenuItem,{href:"/project-admin-dashboard",icon:{name:"AccountGroupShieldAdd"},title:"Users Dashboard"}),(0,O.jsx)(f.MenuItem,{href:"/site-dashboard",icon:{name:"ConnectionNetworkComputers2"},title:"Client Sites"}),(0,O.jsx)(f.MenuItem,{href:"/project-configuration",icon:{name:"SettingsCog"},title:"Project Configuration"}),(0,O.jsx)(f.MenuItem,{href:"/server-config",icon:{name:"ConnectionServerNetwork1"},title:"Server Configuration"}),(0,O.jsx)(f.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,O.jsx)(f.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(1==D.user.role||2==D.user.role||3==D.user.role)&&(0,O.jsxs)(f.MenuContent,{children:[(0,O.jsx)(f.MenuItem,{href:"/",icon:{name:"ViewList"},title:"Project Home Page"}),(0,O.jsx)(f.MenuItem,{href:"/user-dashboard",icon:{name:"ServerEdit"},title:"My Info"}),(0,O.jsx)(f.MenuItem,{href:"/downloads",icon:{name:"ActionsDownload"},title:"Downloads"}),(0,O.jsx)(f.MenuItem,{href:"/logout",icon:{name:"PlaybackStop"},title:"Logout"})]}),(0,O.jsx)(f.MenuFooter,{})]}),(0,O.jsxs)(g,{children:[(0,O.jsx)(d(),{className:"content-header",title:s,children:o}),(0,O.jsx)("div",{className:"content-wrapper",children:n})]})]})}},78455:function(e,t,n){"use strict";n.d(t,{mg:function(){return v},nv:function(){return _}}),n(67294);var r=n(85444);n(40398);var o=n(90878),i=n.n(o),s=n(85893);(0,r.default)(i()).withConfig({displayName:"ErrorMessage__StyledErrorText",componentId:"sc-azomh6-0"})(["display:block;color:",";font-size:",";font-weight:",";position:absolute;"],function(e){return e.theme.colors.red500},function(e){return e.theme.typography.size.small},function(e){return e.theme.typography.weight.bold}),r.default.div.withConfig({displayName:"CheckboxField__CheckboxWrapperStyled",componentId:"sc-1m8bzxk-0"})(["align-self:",";display:",";.checkbox-label-wrapper{display:flex;align-items:flex-start;.checkbox-custom-label{margin-left:",";}}.checkbox-text,.checkbox-custom-label{font-size:",";white-space:break-spaces;}"],function(e){return e.centerVertically?"center":""},function(e){return e.centerVertically?"flex":"block"},function(e){return e.theme.spacing.one},function(e){return e.theme.typography.size.small});var a=n(8307),c=n(4730),l=n(73214),u=n(32754),d=n.n(u),f=n(57299),p=r.default.div.withConfig({displayName:"InfoMessage__FieldHint",componentId:"sc-s0s5lu-0"})(["display:flex;align-items:center;svg{margin-right:",";}"],function(e){return e.theme.spacing.one}),h=function(e){var t=e.children;return(0,s.jsxs)(p,{children:[(0,s.jsx)(f.default,{name:"StatusCircleInformation",size:"small"}),t]})},m=r.default.div.withConfig({displayName:"styles__StyledField",componentId:"sc-1fjauag-0"})(["> div{margin-bottom:0px;}margin-bottom:",";"],function(e){return e.theme.spacing.four}),g=["name","disabled","options","placeholder","info","wrapperClass"];function j(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function b(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?j(Object(n),!0).forEach(function(t){(0,a.Z)(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):j(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var v=function(e){var t=e.name,n=e.disabled,r=e.options,o=e.placeholder,i=e.info,a=e.wrapperClass,u=void 0===a?"":a,f=(0,c.Z)(e,g);return(0,s.jsx)(l.gN,{name:t,children:function(e){var a=e.form,c=!!a.errors[t]&&(a.submitCount>0||a.touched[t]);return(0,s.jsxs)(m,{className:u,children:[(0,s.jsx)(d(),b(b({},f),{},{disabled:!!n||a.isSubmitting,name:t,onBlur:function(){return a.setFieldTouched(t,!0)},onChange:function(e){return a.setFieldValue(t,e)},options:r,placeholder:o,valid:!c,validationMessage:c?a.errors[t]:void 0,value:null==a?void 0:a.values[t]})),i&&(0,s.jsx)(h,{children:i})]})}})},x=n(24777),y=n.n(x),w=["className","name","info","label","disabled","placeholder"];function P(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function S(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?P(Object(n),!0).forEach(function(t){(0,a.Z)(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):P(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var _=function(e){var t=e.className,n=e.name,r=e.info,o=e.label,i=e.disabled,a=e.placeholder,u=(0,c.Z)(e,w);return(0,s.jsx)(l.gN,{name:n,children:function(e){var c=e.form,l=!!c.errors[n]&&(c.submitCount>0||c.touched[n]);return(0,s.jsxs)(m,{children:[(0,s.jsx)(y(),S(S({},u),{},{className:t,disabled:!!i||c.isSubmitting,label:o,name:n,onBlur:c.handleBlur,onChange:c.handleChange,placeholder:a,valid:!l,validationMessage:l?c.errors[n]:void 0,value:null==c?void 0:c.values[n]})),r&&(0,s.jsx)(h,{children:r})]})}})}},89002:function(e,t,n){"use strict";n.d(t,{P:function(){return l},X:function(){return c}});var r=n(85444),o=n(36578),i=n.n(o),s=n(3159),a=n.n(s),c=(0,r.default)(a()).withConfig({displayName:"form-page__StyledFormExample",componentId:"sc-rfrcq8-0"})([".bottom{display:flex;gap:",";}.zero-left{margin-left:0;}.zero-right{margin-right:0;}"],function(e){return e.theme.spacing.four}),l=(0,r.default)(i()).withConfig({displayName:"form-page__StyledBanner",componentId:"sc-rfrcq8-1"})(["margin-bottom:1rem;"])},35921:function(e,t,n){"use strict";n.r(t);var r=n(8307),o=n(73214),i=n(74231),s=n(5801),a=n.n(s),c=n(90878),l=n.n(c),u=n(12976),d=n(89002),f=n(67294),p=n(78455),h=n(14682),m=n(11163),g=n(35827),j=n.n(g),b=n(85893);function v(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function x(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?v(Object(n),!0).forEach(function(t){(0,r.Z)(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):v(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var y=i.Ry().shape({short_name:i.Z_().min(1,"Too Short!").max(16,"Too Long!"),description:i.Z_(),app_location:i.Z_()});t.default=function(){var e=(0,m.useRouter)().push,t=(0,f.useState)(""),n=t[0],r=t[1],i=(0,f.useState)(),s=i[0],c=i[1];return(0,f.useEffect)(function(){h.y.getProject().then(function(e){c(e.project)})},[]),(0,b.jsx)(b.Fragment,{children:(0,b.jsx)(u.Z,{title:"Project Configuration:",children:s?(0,b.jsx)(o.J9,{initialValues:s,onSubmit:function(t,n){h.y.patchProject({short_name:t.short_name,title:t.title,description:t.description,app_location:t.app_location,starting_date:t.starting_date,end_date:t.end_date,frozen:!1}).then(function(e){c(e.project),setTimeout(function(){n.setSubmitting(!1),n.resetForm({values:e.project})},600)}).catch(function(t){console.log(t),e("/")})},validationSchema:y,children:function(t){return(0,b.jsx)(b.Fragment,{children:(0,b.jsxs)("div",{style:{minHeight:"835px"},children:[t.values.frozen&&(0,b.jsx)(d.P,{status:"warning",rounded:!0,children:"This project has been frozen. Project values are no longer editable."}),(0,b.jsxs)(d.X,{loading:t.isSubmitting,title:"Project Values",children:[(0,b.jsx)(l(),{tag:"label",textStyle:"p1",children:"Set the values for the information of the FL project."}),(0,b.jsx)(p.nv,{disabled:t.isSubmitting||t.values.frozen,label:"Project Short Name (maximum 16 characters, at top left, and used in certs)",name:"short_name"}),(0,b.jsx)(p.nv,{disabled:t.isSubmitting||t.values.frozen,label:"Title (on home page)",name:"title"}),(0,b.jsx)(p.nv,{inputType:"multiLine",disabled:t.isSubmitting||t.values.frozen,label:"Project Description",name:"description"}),(0,b.jsx)(p.nv,{label:"Docker Download Link",name:"app_location",placeholder:"",disabled:t.isSubmitting||t.values.frozen}),(0,b.jsxs)("div",{className:"bottom",children:[""===t.values.starting_date||"starting_date"===n?(0,b.jsx)(o.gN,{as:j(),className:"zero-left zero-right",disabled:t.isSubmitting||t.values.frozen,label:"Project Start Date",name:"starting_date",pattern:t.errors.starting_date,onChange:function(e){e>new Date(t.values.end_date?t.values.end_date:new Date(864e13))?alert("Please select a Project Start Date before the Project End Date."):(t.setFieldValue("starting_date",e.toDateString()),r(""))}}):(0,b.jsx)(b.Fragment,{children:(0,b.jsxs)("div",{style:{display:"inline-block",width:"240px"},children:[(0,b.jsx)("div",{children:(0,b.jsx)("p",{style:{margin:"0 0 0.25rem"},children:"Project Start Date"})}),(0,b.jsx)("br",{}),(0,b.jsx)("div",{style:{position:"relative",top:"-12px",right:"-16px"},onClick:function(){t.values.frozen||r("starting_date")},children:t.values.starting_date})]})}),""===t.values.end_date||"end_date"===n?(0,b.jsx)(o.gN,{as:j(),className:"zero-left zero-right",disabled:t.isSubmitting||t.values.frozen,label:"Project End Date",name:"end_date",pattern:t.errors.end_date,onChange:function(e){e<new Date(t.values.starting_date?t.values.starting_date:0)?alert("Please select a Project End Date after the Project Start Date."):(t.setFieldValue("end_date",e.toDateString()),r(""))}}):(0,b.jsx)(b.Fragment,{children:(0,b.jsxs)("div",{style:{display:"inline-block",width:"240px"},children:[(0,b.jsx)("div",{children:(0,b.jsx)("p",{style:{margin:"0 0 0.25rem"},children:"Project End Date"})}),(0,b.jsx)("br",{}),(0,b.jsx)("div",{style:{position:"relative",top:"-12px",right:"-16px"},onClick:function(){t.values.frozen||r("end_date")},children:t.values.end_date})]})}),t.values.public?(0,b.jsx)(a(),{disabled:!0,onClick:function(){h.y.patchProject({public:!1}).then(function(e){c(e.project)}).catch(function(t){console.log(t),e("/")}),t.values.public=!1,c(x(x({},s),{},{public:!1}))},children:"Project is Public"}):(0,b.jsx)(a(),{onClick:function(){h.y.patchProject({public:!0}).then(function(e){c(e.project)}).catch(function(t){console.log(t),e("/")}),t.values.public=!0,c(x(x({},s),{},{public:!0}))},children:"Make Project Public"}),(0,b.jsx)("div",{style:{marginTop:"18px"},children:"If the project is public, user signup is enabled."})]}),(0,b.jsx)("br",{}),(0,b.jsx)(a(),{disabled:!t.dirty||!t.isValid||t.values.frozen,onClick:function(){return t.handleSubmit()},children:"Save"}),(0,b.jsx)("br",{}),!t.values.frozen&&(0,b.jsx)(d.P,{status:"info",rounded:!0,children:"After setting all the values for the project configuartion, application, and server configuration, click Freeze Project on the Project Home page to freeze all the values and allow downloads of the project artifacts."})]})]})})}}):(0,b.jsx)("span",{children:"loading..."})})})}},40203:function(e,t,n){"use strict";n.d(t,{Gg:function(){return s},KY:function(){return o},a5:function(){return i}});var r={user:{email:"",token:"",id:-1,role:0},status:"unauthenticated"};function o(e,t,n,o,i){var s=arguments.length>5&&void 0!==arguments[5]&&arguments[5];return r={user:{email:e,token:t,id:n,role:o,org:i},expired:s,status:0==o?"unauthenticated":"authenticated"},localStorage.setItem("session",JSON.stringify(r)),r}function i(e){return r={user:{email:r.user.email,token:r.user.token,id:r.user.id,role:e,org:r.user.org},expired:r.expired,status:r.status},localStorage.setItem("session",JSON.stringify(r)),r}function s(){var e=localStorage.getItem("session");return null!=e&&(r=JSON.parse(e)),r}},71068:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/project-configuration",function(){return n(35921)}])},84506:function(e){"use strict";e.exports=JSON.parse('{"projectname":"New FL Project","demo":false,"url_root":"/nvflare-dashboard","arraydata":[{"name":"itemone"},{"name":"itemtwo"},{"name":"itemthree"}]}')}},function(e){e.O(0,[234,787,403,888,774,179],function(){return e(e.s=71068)}),_N_E=e.O()}]);