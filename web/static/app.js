import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const state = {
  jobs: [],
  activeJob: null,
  viewerData: null,
  meshVisible: true,
  voxelsVisible: false,
  surfaceVisible: true,
  featuresVisible: true,
  meshObject: null,
  voxelObject: null,
  surfaceObject: null,
  featureObject: null,
  approachObject: null,
};

const els = {
  apiState: document.querySelector("#api-state"),
  jobList: document.querySelector("#job-list"),
  activeJob: document.querySelector("#active-job"),
  recommendation: document.querySelector("#recommendation"),
  summaryGrid: document.querySelector("#summary-grid"),
  reviewCodes: document.querySelector("#review-codes"),
  phaseList: document.querySelector("#phase-list"),
  simulationPanel: document.querySelector("#simulation-panel"),
  operationList: document.querySelector("#operation-list"),
  uploadForm: document.querySelector("#upload-form"),
  refreshJobs: document.querySelector("#refresh-jobs"),
  toggleMesh: document.querySelector("#toggle-mesh"),
  toggleVoxels: document.querySelector("#toggle-voxels"),
  toggleSurface: document.querySelector("#toggle-surface"),
  toggleFeatures: document.querySelector("#toggle-features"),
  viewer: document.querySelector("#viewer"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111316);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
camera.position.set(120, -150, 110);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
els.viewer.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x222831, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(80, -90, 140);
scene.add(keyLight);
const grid = new THREE.GridHelper(180, 18, 0x2f3942, 0x232a31);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

function resize() {
  const rect = els.viewer.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function text(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function clearObject(object) {
  if (!object) return;
  scene.remove(object);
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach((m) => m.dispose?.());
    } else {
      child.material?.dispose?.();
    }
  });
}

function pointsToCloud(pointsData, color, size = 1.4, opacity = 0.82) {
  const points = pointsData?.viewer_points || pointsData?.points || [];
  const positions = new Float32Array(points.length * 3);
  points.forEach((point, i) => {
    positions[i * 3 + 0] = point[0];
    positions[i * 3 + 1] = point[1];
    positions[i * 3 + 2] = point[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color,
    size,
    sizeAttenuation: true,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
}

function createFeatureOverlays(overlays = []) {
  const group = new THREE.Group();
  overlays.forEach((overlay) => {
    const color = new THREE.Color(overlay.color || "#f2b84b");
    const size = overlay.bbox_size || [1, 1, 1];
    const center = overlay.bbox_center || [0, 0, 0];
    const boxGeometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
    const edges = new THREE.EdgesGeometry(boxGeometry);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }),
    );
    line.position.set(center[0], center[1], center[2]);
    line.userData = overlay;
    group.add(line);

    const centroid = overlay.centroid || center;
    const markerGeometry = new THREE.SphereGeometry(Math.max(0.8, Math.min(...size) * 0.08), 10, 8);
    const marker = new THREE.Mesh(
      markerGeometry,
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, roughness: 0.4 }),
    );
    marker.position.set(centroid[0], centroid[1], centroid[2]);
    marker.userData = overlay;
    group.add(marker);
  });
  group.visible = state.featuresVisible;
  return group;
}

function createApproachOverlay(setupOverlay) {
  if (!setupOverlay?.arrow_start || !setupOverlay?.arrow_end) return null;
  const start = new THREE.Vector3(...setupOverlay.arrow_start);
  const end = new THREE.Vector3(...setupOverlay.arrow_end);
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  if (length <= 0) return null;
  const group = new THREE.Group();
  const arrow = new THREE.ArrowHelper(direction.normalize(), start, length, 0xf2b84b, length * 0.22, length * 0.1);
  group.add(arrow);
  return group;
}

async function loadMesh(jobId, meshUrl) {
  if (!meshUrl) return null;
  return new Promise((resolve, reject) => {
    new STLLoader().load(
      meshUrl,
      (geometry) => {
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          color: 0x9fb5c4,
          metalness: 0.1,
          roughness: 0.55,
          transparent: true,
          opacity: 0.42,
          depthWrite: false,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `mesh-${jobId}`;
        resolve(mesh);
      },
      undefined,
      reject,
    );
  });
}

async function renderViewer(data) {
  clearObject(state.meshObject);
  clearObject(state.voxelObject);
  clearObject(state.surfaceObject);
  clearObject(state.featureObject);
  clearObject(state.approachObject);
  state.meshObject = null;
  state.voxelObject = null;
  state.surfaceObject = null;
  state.featureObject = null;
  state.approachObject = null;

  state.voxelObject = pointsToCloud(data.voxel_points, 0x54b7a7, 0.8, 0.22);
  state.voxelObject.visible = state.voxelsVisible;
  scene.add(state.voxelObject);

  state.surfaceObject = pointsToCloud(data.surface_points, 0x65d6c6, 1.25, 0.88);
  state.surfaceObject.visible = state.surfaceVisible;
  scene.add(state.surfaceObject);

  state.featureObject = createFeatureOverlays(data.feature_overlays || []);
  scene.add(state.featureObject);

  state.approachObject = createApproachOverlay(data.setup_overlay);
  if (state.approachObject) scene.add(state.approachObject);

  state.meshObject = await loadMesh(data.job_id, data.mesh_url);
  if (state.meshObject) {
    state.meshObject.visible = state.meshVisible;
    scene.add(state.meshObject);
  }

  const bbox = new THREE.Box3().setFromObject(state.meshObject || state.voxelObject || state.surfaceObject || state.featureObject);
  if (!bbox.isEmpty()) {
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bbox.getCenter(center);
    bbox.getSize(size);
    controls.target.copy(center);
    const radius = Math.max(size.x, size.y, size.z, 60);
    camera.position.set(center.x + radius * 0.9, center.y - radius * 1.2, center.z + radius * 0.8);
    camera.near = Math.max(0.1, radius / 1000);
    camera.far = radius * 20;
    camera.updateProjectionMatrix();
  }
}

function renderJobs() {
  els.jobList.innerHTML = "";
  if (!state.jobs.length) {
    els.jobList.innerHTML = '<div class="empty">No jobs found.</div>';
    return;
  }
  state.jobs.forEach((job) => {
    const node = document.createElement("div");
    node.className = `job-item ${state.activeJob === job.job_id ? "active" : ""}`;
    node.innerHTML = `
      <div class="job-title"><strong>${job.job_id}</strong><span>${text(job.recommendation, job.state)}</span></div>
      <div class="meta">${text(job.setup_count)} setup · ${text(job.axis_requirement)} axis · ${text(job.operation_count)} ops</div>
    `;
    node.addEventListener("click", () => loadJob(job.job_id));
    els.jobList.appendChild(node);
  });
}

function renderSummary(data) {
  const quote = data.quotation || {};
  const setup = data.setup_analysis || {};
  const plan = data.process_plan || {};
  const time = data.time_estimate || {};
  const rec = text(quote.recommendation, "UNKNOWN");
  els.recommendation.textContent = rec;
  els.recommendation.className = `recommendation ${rec.toLowerCase()}`;
  const cost = quote.estimated_cost?.total;
  const currency = quote.estimated_cost?.currency || "";
  const cells = [
    ["Setups", setup.setup_count ?? plan.setup_count],
    ["Axes", setup.axis_requirement ?? plan.axis_requirement],
    ["Operations", plan.operation_count],
    ["Time", time.total_time_min ? `${time.total_time_min.toFixed?.(1) || time.total_time_min} min` : null],
    ["Cost", cost ? `${currency} ${Number(cost).toFixed(2)}` : null],
    ["Tool Reach", setup.tool_reach_compatible === false ? "Review" : "OK"],
  ];
  els.summaryGrid.innerHTML = cells
    .map(([label, value]) => `<div class="summary-cell"><span class="label">${label}</span><strong>${text(value)}</strong></div>`)
    .join("");
}

function renderReviewCodes(data) {
  const codes = data.quotation?.review_codes || data.process_plan?.review_codes || data.setup_analysis?.review_codes || [];
  els.reviewCodes.innerHTML = codes.length
    ? codes.map((code) => `<span class="pill">${code}</span>`).join("")
    : '<div class="empty">No review codes.</div>';
}

function phaseRows(data) {
  const setup = data.setup_analysis || {};
  const plan = data.process_plan || {};
  const quote = data.quotation || {};
  return [
    ["Phase 1", "Geometry", data.metadata?.bounding_box_mm ? JSON.stringify(data.metadata.bounding_box_mm) : "No metadata"],
    ["Phase 2", "Features", `${data.features?.feature_count ?? 0} detected`],
    ["Phase 2c", "Instances", `${data.feature_instances?.instance_count ?? 0} localised`],
    ["Phase 3", "Setup", `${setup.setup_mode || "—"} · reach ${setup.tool_reach_compatible === false ? "review" : "ok"}`],
    ["Phase 4", "Plan", `${plan.operation_count ?? 0} operations`],
    ["Phase 5", "Time", `${data.time_estimate?.total_time_min ?? "—"} min`],
    ["Phase 6", "Quote", quote.recommendation || "—"],
  ];
}

function renderPhases(data) {
  els.phaseList.innerHTML = phaseRows(data)
    .map(
      ([phase, title, detail]) => `
        <div class="phase-item">
          <div class="phase-title"><strong>${phase}</strong><span>${title}</span></div>
          <div class="meta">${detail}</div>
        </div>
      `,
    )
    .join("");
}

function renderSimulation(data) {
  const sim = data.simulation_input || {};
  const readiness = sim.readiness || {};
  const setup = sim.setup || data.setup_analysis || {};
  const operations = sim.operations || [];
  const warnings = [...(readiness.errors || []), ...(readiness.warnings || [])];
  const reviewCodes = readiness.review_codes || [];
  const rec = readiness.recommendation || "MISSING";
  els.simulationPanel.innerHTML = `
    <div class="simulation-state ${rec.toLowerCase()}">
      <strong>${rec}</strong>
      <span>${readiness.ready_for_simulation ? "ready for simulation" : "needs review"}</span>
    </div>
    <div class="summary-grid compact">
      <div class="summary-cell"><span class="label">Setup</span><strong>${text(setup.setup_count)} · ${text(setup.setups?.[0]?.approach_direction || setup.setup_mode)}</strong></div>
      <div class="summary-cell"><span class="label">Operations</span><strong>${operations.length || data.process_plan?.operation_count || 0}</strong></div>
      <div class="summary-cell"><span class="label">Instances</span><strong>${data.feature_instances?.instance_count ?? 0}</strong></div>
      <div class="summary-cell"><span class="label">2.5D</span><strong>${setup.two_point_five_d_compatible === false ? "Review" : "OK"}</strong></div>
    </div>
    <div class="handoff-flags">
      ${
        reviewCodes.length
          ? reviewCodes.map((code) => `<span class="pill">${code}</span>`).join("")
          : '<div class="empty">No simulation review codes.</div>'
      }
    </div>
    ${
      warnings.length
        ? `<div class="handoff-warnings">${warnings.map((warning) => `<div>${warning}</div>`).join("")}</div>`
        : ""
    }
  `;
}

function renderOperations(data) {
  const operations = data.simulation_input?.operations || data.process_plan?.operations || [];
  els.operationList.innerHTML = operations.length
    ? operations
        .slice(0, 40)
        .map(
          (op) => `
          <div class="operation-item">
            <div class="operation-title"><strong>${op.operation_id || op.step}. ${op.operation_type}</strong><span>${op.approach_direction}</span></div>
            <div class="meta">${op.feature_type} · ${op.tool_type} · ${op.phase}</div>
            <div class="meta">instance ${text(op.feature_instance_id, "n/a")} · tool Ø ${text(op.tool_diameter_mm, "n/a")} mm · depth ${text(op.cut_depth_mm, "n/a")} mm</div>
            ${op.requires_review ? '<div class="operation-review">Needs review</div>' : ""}
          </div>
        `,
        )
        .join("")
    : '<div class="empty">No operations available.</div>';
}

async function refreshJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  renderJobs();
  if (!state.activeJob && state.jobs[0]) {
    await loadJob(state.jobs[0].job_id);
  }
}

async function loadJob(jobId) {
  state.activeJob = jobId;
  els.activeJob.textContent = jobId;
  renderJobs();
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}/viewer-data`);
  state.viewerData = data;
  renderSummary(data);
  renderReviewCodes(data);
  renderPhases(data);
  renderSimulation(data);
  renderOperations(data);
  await renderViewer(data);
}

async function submitJob(event) {
  event.preventDefault();
  const file = document.querySelector("#step-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("step_file", file);
  form.append("factory_profile", document.querySelector("#factory-profile").value);
  form.append("resolution", document.querySelector("#resolution").value);
  form.append("confidence", document.querySelector("#confidence").value);
  const model = document.querySelector("#model").value.trim();
  if (model) form.append("model", model);
  const result = await api("/api/jobs", { method: "POST", body: form });
  await refreshJobs();
  await loadJob(result.job_id);
}

function bindToggle(button, key, objectKey) {
  button.addEventListener("click", () => {
    state[key] = !state[key];
    button.classList.toggle("active", state[key]);
    if (state[objectKey]) state[objectKey].visible = state[key];
  });
}

els.refreshJobs.addEventListener("click", refreshJobs);
els.uploadForm.addEventListener("submit", submitJob);
bindToggle(els.toggleMesh, "meshVisible", "meshObject");
bindToggle(els.toggleVoxels, "voxelsVisible", "voxelObject");
bindToggle(els.toggleSurface, "surfaceVisible", "surfaceObject");
bindToggle(els.toggleFeatures, "featuresVisible", "featureObject");

api("/api/health")
  .then(() => {
    els.apiState.textContent = "API online";
    return refreshJobs();
  })
  .catch((error) => {
    els.apiState.textContent = "API error";
    console.error(error);
  });
