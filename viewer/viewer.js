const viewer = new Cesium.Viewer("cesiumContainer", {
    shouldAnimate: true,
});

// datetime-local inputs display local time; send UTC ISO timestamps to Flask.
function localInputValue(date) {
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000)
        .toISOString().slice(0, 16);
}

const now = new Date();
document.getElementById("start").value = localInputValue(now);
document.getElementById("end").value = localInputValue(new Date(now.getTime() + 2 * 3600000));
let activeDataSource;
const satelliteColors = new Map();

function satelliteColor(name) {
    if (!satelliteColors.has(name)) {
        // Space successive hues apart and keep each name's color on reload.
        const hue = (0.55 + satelliteColors.size * 0.61803398875) % 1;
        satelliteColors.set(name, Cesium.Color.fromHsl(hue, 0.85, 0.6));
    }
    return satelliteColors.get(name);
}

document.getElementById("controls").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = document.getElementById("load");
    const status = document.getElementById("status");
    button.disabled = true;
    status.textContent = "Fetching elements and propagating trajectories…";
    try {
        const response = await fetch("/propagate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                objects: document.getElementById("objects").value
                    .split(",").map(name => name.trim()).filter(Boolean),
                start_date: new Date(document.getElementById("start").value).toISOString(),
                end_date: new Date(document.getElementById("end").value).toISOString(),
            }),
        });
        if (!response.ok) {
            const detail = await response.json().catch(() => ({}));
            throw new Error(detail.error || `Server returned HTTP ${response.status}`);
        }
        const czml = await response.json();
        const dataSource = await Cesium.CzmlDataSource.load(czml);
        for (const entity of dataSource.entities.values) {
            const color = satelliteColor(entity.name || entity.id);
            if (entity.point) {
                entity.point.color = color;
            }
            if (entity.path) {
                entity.path.material = new Cesium.ColorMaterialProperty(color);
            }
            entity.label = new Cesium.LabelGraphics({
                text: entity.name || entity.id,
                font: "14px sans-serif",
                fillColor: Cesium.Color.WHITE,
                showBackground: true,
                backgroundColor: Cesium.Color.BLACK.withAlpha(0.75),
                pixelOffset: new Cesium.Cartesian2(14, 0),
                horizontalOrigin: Cesium.HorizontalOrigin.LEFT,
                verticalOrigin: Cesium.VerticalOrigin.CENTER,
            });
        }
        await viewer.dataSources.add(dataSource);
        if (activeDataSource) {
            viewer.dataSources.remove(activeDataSource, true);
        }
        activeDataSource = dataSource;
        viewer.clockTrackedDataSource = dataSource;
        viewer.clock.currentTime = dataSource.clock.startTime.clone();
        viewer.clock.multiplier = 60;
        viewer.clock.shouldAnimate = true;
        status.textContent = `Displaying ${dataSource.entities.values.length} satellite(s).`;
    } catch (error) {
        status.textContent = `Could not load trajectories: ${error.message}`;
    } finally {
        button.disabled = false;
    }
});
