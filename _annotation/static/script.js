//let polygons = [{points: [], label: ''}]; // Initialize with one polygon object

let currentPolygonIndex = polygons.length-1;

function isClickedInsidePolygon(polygon, mouseX, mouseY) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        let xi = polygon[i].x, yi = polygon[i].y;
        let xj = polygon[j].x, yj = polygon[j].y;

        let intersect = ((yi > mouseY) != (yj > mouseY))
            && (mouseX < (xj - xi) * (mouseY - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}


document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const image = new Image();
    image.src = 'static/1.jpg'; // Make sure the path to your image is correct

    let scale = 1;
    const scaleMultiplier = 0.1;
    let imgPosition = { x: 0, y: 0 };
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };


    let isPointDragging = false;
    let draggedPoint = { polygonIndex: -1, pointIndex: -1 };

    image.onload = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        drawImage();
    };

    function drawImage() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, imgPosition.x, imgPosition.y, image.width * scale, image.height * scale);
        drawPolygons();
    }

 function drawPolygons() {
    polygons.forEach((polygonObj, polygonIdx) => {
        const polygon = polygonObj.points;
        if (polygon.length > 1) {
            ctx.beginPath();
            ctx.moveTo((polygon[0].x * scale) + imgPosition.x, (polygon[0].y * scale) + imgPosition.y);
            polygon.forEach(point => {
                ctx.lineTo((point.x * scale) + imgPosition.x, (point.y * scale) + imgPosition.y);
            });
            ctx.closePath();
            if( polygonObj.label == "") {
                ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
            } else if(polygonObj.label == "a") {
                ctx.fillStyle = 'rgba(0, 155, 0, 0.5)';            
            } else {
                ctx.fillStyle = 'rgba(155, 0, 0, 0.5)';            
            }
            
            ctx.fill();
            ctx.stroke();

            // Only draw points if the polygon is currently selected
            if (polygonIdx === currentPolygonIndex) {
                polygon.forEach((point, pointIdx) => {
                    ctx.beginPath();
                    ctx.arc((point.x * scale) + imgPosition.x, (point.y * scale) + imgPosition.y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = isPointDragging && draggedPoint.polygonIndex === polygonIdx && draggedPoint.pointIndex === pointIdx ? 'red' : 'black';
                    ctx.fill();
                });
            }
            
            // Draw label if it exists
            if (polygonObj.label) {
                ctx.font = '16px Arial';
                ctx.fillStyle = 'black';
                ctx.fillText(polygonObj.label, (polygon[0].x * scale) + imgPosition.x, (polygon[0].y * scale) + imgPosition.y - 10);
            }
        }
    });
}



    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const mousePosition = { x: e.clientX, y: e.clientY };
        const rect = canvas.getBoundingClientRect();

        const mousePositionRelativeToCanvas = {
            x: mousePosition.x - rect.left,
            y: mousePosition.y - rect.top,
        };

        const scaleFactor = Math.exp(e.deltaY * scaleMultiplier * -0.01);
        const newScale = scale * scaleFactor;

        const scaleChange = newScale - scale;
        imgPosition.x -= (mousePositionRelativeToCanvas.x - imgPosition.x) * (scaleChange / scale);
        imgPosition.y -= (mousePositionRelativeToCanvas.y - imgPosition.y) * (scaleChange / scale);
        scale = newScale;

        drawImage();
    });

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 2) {
            e.preventDefault();
            isDragging = true;
            dragStart.x = e.clientX - imgPosition.x;
            dragStart.y = e.clientY - imgPosition.y;
        } else if (e.button === 0) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left - imgPosition.x) / scale;
            const mouseY = (e.clientY - rect.top - imgPosition.y) / scale;

            let pointFound = false;
            polygons.forEach((polygonObj, polygonIdx) => {
              if (polygonIdx === currentPolygonIndex) {    
                  polygonObj.points.forEach((point, pointIndex) => { // Corrected to access .points
                      const distance = Math.sqrt((point.x - mouseX) ** 2 + (point.y - mouseY) ** 2);
                      if (distance < 5 / scale) {
                          isPointDragging = true;
                          draggedPoint = { polygonIndex: polygonIdx, pointIndex }; // Ensure correct reference to index
                          pointFound = true;
                          e.preventDefault();
                          return;
                      }
                  });
                  if (pointFound) return;
               }
            });
            isPolygon = false;            
            if (!pointFound && !isDragging) {
              // Check if the click is inside any polygon. This requires a more complex algorithm,
              // such as ray-casting, which is not fully implemented here.
              // Placeholder for checking click inside a polygon
              
              polygons.forEach((polygonObj, polygonIdx) => {
                  // Implement point-in-polygon check here, set clickedInsidePolygon = true if inside
                  let clickedInsidePolygon = isClickedInsidePolygon(polygonObj.points, mouseX, mouseY);
                  if (clickedInsidePolygon) {
                      currentPolygonIndex = polygonIdx; // Select the polygon if clicked inside
                      isPolygon = true;
                      drawImage();
                      return; // Stop checking other polygons once we find a match
                  }
              });
              


            if (!pointFound && !isDragging && !isPolygon) {
                if(polygons.length - 1 != currentPolygonIndex) {
                  polygons.push({points: [], label: ''}); // Initialize new polygon
                  currentPolygonIndex = polygons.length - 1;                   
                }
                polygons[currentPolygonIndex].points.push({ x: mouseX, y: mouseY });
                drawImage();
            }
          }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDragging) {
            e.preventDefault();
            imgPosition.x = e.clientX - dragStart.x;
            imgPosition.y = e.clientY - dragStart.y;
            drawImage();
        } else if (isPointDragging) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left - imgPosition.x) / scale;
            const mouseY = (e.clientY - rect.top - imgPosition.y) / scale;
    
            // Corrected path to update the dragged point's position
            polygons[draggedPoint.polygonIndex].points[draggedPoint.pointIndex] = { x: mouseX, y: mouseY };
            drawImage();
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
            if (isPointDragging) {
              isPointDragging = false;
              draggedPoint = { polygonIndex: -1, pointIndex: -1 };
              drawImage();
            }
       } else if (e.button === 2) { // Right mouse button
           isDragging = false;
       }
    });


    canvas.addEventListener('dblclick', (e) => {
        if (polygons[currentPolygonIndex].length > 2) {
            currentPolygonIndex++;
            polygons.push([]);
        }
        drawImage();
    });

    canvas.oncontextmenu = (e) => {
        e.preventDefault();
        e.stopPropagation();
        return false;
    };
    
    document.addEventListener('keydown', (e) => {
        if (e.key === "a" && polygons[currentPolygonIndex].points.length > 2) { // Ensure valid polygon
            polygons[currentPolygonIndex].label = 'a'; // Assign label "a"
            currentPolygonIndex = polygons.length; // Move to next polygon
            polygons.push({points: [], label: ''}); // Initialize new polygon
            drawImage(); // Redraw to show updates
        } else if (e.key === "n" && polygons[currentPolygonIndex].points.length > 2) { // Ensure valid polygon
            polygons[currentPolygonIndex].label = 'n'; // Assign label "a"
            currentPolygonIndex = polygons.length; // Move to next polygon
            polygons.push({points: [], label: ''}); // Initialize new polygon
            drawImage(); // Redraw to show updates
        } else if (e.key === "Delete" || e.key === "Backspace") { // Add check for delete key
            if (currentPolygonIndex !== -1 && polygons.length >= 1) { // Ensure there is a selected polygon to delete and at least one polygon remains
                //alert("AAAAA");
                polygons[currentPolygonIndex].points.length = 0;
                polygons[currentPolygonIndex].label = "";
                drawImage(); // Redraw to reflect the deletion
            }
        }
    });

});

function submitPolygons() {
    const formData = JSON.stringify(polygons); // Assuming 'polygons' is already defined

    fetch('/submit-polygons', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: formData,
    })
    .then(response => response.json())
    .then(data => console.log('Success:', data))
    .catch((error) => {
        console.error('Error:', error);
        //alert(formData); // Consider adjusting or removing alerts in production code
    });
}

document.getElementById('changeUrlBtn').addEventListener('click', function() {
    // Change the URL without reloading the page
    window.location.href = '/predict';
});

