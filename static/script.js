const sidebarToggle = document.getElementById("sidebar-toggle");
      const sidebar = document.getElementById("sidebar");
      const mainContent = document.getElementById("main-content");
      const videoFeedOutsideModal =
        document.getElementById("video_feed_outside");
      let modal = null; // Reference to the modal

      // Toggle sidebar and main content classes
      sidebarToggle.addEventListener("click", () => {
        sidebar.classList.toggle("active");
        mainContent.classList.toggle("adjusted");
      });

      function getOrCreateModal() {
        if (!modal) {
          modal = document.createElement("div");
          modal.className = "modal";
          modal.innerHTML = `
                  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                    <div class="bg-gray-800 p-4 rounded-lg">
                      <h2 class="text-xl font-bold mb-4">Webcam Feed</h2>
                      <div class="video-container">
                        <img id="video_feed" src="" alt="Video Feed">
                      </div>
                      <div class="mt-2">
                        <button id="close" class="bg-red-600 px-4 py-2 rounded-lg">Close</button>
                      </div>
                    </div>
                  </div>
                `;
          document.body.appendChild(modal);

          // Close button event listener
          modal
            .querySelector("#close")
            .addEventListener("click", () => closeModal());
        }
        return modal;
      }
      function openVideoWindow() {
        const modal = getOrCreateModal();
        updateVideoFeedSource(
          modal.querySelector("#video_feed"),
          "{{ url_for('video_feed') }}"
        );
        modal.style.display = "block"; // Show the modal
      }

      // Close modal function
      function closeModal() {
        if (modal) {
          updateVideoFeedSource(modal.querySelector("#video_feed"), "");
          modal.style.display = "none"; // Hide the modal
        }
      }
      // Update video feed source
      function updateVideoFeedSource(videoElement, source) {
        if (videoElement) {
          videoElement.src = source;
        }
      }

      // Initialize outside video feed
      if (videoFeedOutsideModal) {
        updateVideoFeedSource(
          videoFeedOutsideModal,
          "{{ url_for('video_feed') }}"
        );
      }

      // Expose openVideoWindow globally if needed
      window.openVideoWindow = openVideoWindow;