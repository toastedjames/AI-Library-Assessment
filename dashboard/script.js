async function loadDashboard() {

    const response = await fetch(
        "dashboard_data.json"
    );

    const data = await response.json();


    renderKPIs(data);

    renderDimensions(data);

    renderStudentLevels(data);

    renderThemes(data);

    renderSentiment(data);

    renderModels(data);

}


/* ---------------------------------------------------------
   KPI cards
--------------------------------------------------------- */

function renderKPIs(data) {

    const overview = data.overview;


    document.getElementById(
        "totalResponses"
    ).textContent =
        overview.total_responses;


    document.getElementById(
        "satisfaction"
    ).textContent =
        overview.overall_satisfaction.toFixed(2);


    document.getElementById(
        "nlpAccuracy"
    ).textContent =
        formatPercent(
            overview.nlp_accuracy
        );


    document.getElementById(
        "aiAccuracy"
    ).textContent =
        formatPercent(
            overview.ai_v2_accuracy
        );


    document.getElementById(
        "aiReliability"
    ).textContent =
        formatPercent(
            overview.ai_v2_reliability
        );

}


/* ---------------------------------------------------------
   Assessment dimensions
--------------------------------------------------------- */

function renderDimensions(data) {

    const overview = data.overview;


    const dimensions = [

        {
            label: "Digital Resources",
            value:
                overview.digital_resources
        },

        {
            label: "Research Support",
            value:
                overview.research_support
        },

        {
            label: "Study Space",
            value:
                overview.study_space
        },

        {
            label: "Overall Satisfaction",
            value:
                overview.overall_satisfaction
        },

    ];


    const max = 5;


    const container =
        document.getElementById(
            "dimensionChart"
        );


    container.innerHTML =
        dimensions.map(
            item => {

                const width =
                    (item.value / max)
                    * 100;


                return `
                    <div class="bar-row">

                        <span class="bar-label">
                            ${item.label}
                        </span>

                        <div class="bar-track">

                            <div
                                class="bar-fill"
                                style="width:${width}%"
                            ></div>

                        </div>

                        <span class="bar-value">
                            ${item.value.toFixed(2)}
                        </span>

                    </div>
                `;

            }
        ).join("");

}


/* ---------------------------------------------------------
   Student levels
--------------------------------------------------------- */

function renderStudentLevels(data) {

    const rows =
        data.student_level;


    const container =
        document.getElementById(
            "levelChart"
        );


    const max = 5;


    container.innerHTML =
        rows.map(
            row => {

                const width =
                    (
                        row.satisfaction_score
                        / max
                    ) * 100;


                return `
                    <div class="bar-row">

                        <span class="bar-label">
                            ${row.student_level}
                        </span>

                        <div class="bar-track">

                            <div
                                class="bar-fill"
                                style="width:${width}%"
                            ></div>

                        </div>

                        <span class="bar-value">
                            ${row.satisfaction_score.toFixed(2)}
                        </span>

                    </div>
                `;

            }
        ).join("");

}


/* ---------------------------------------------------------
   Themes
--------------------------------------------------------- */

function renderThemes(data) {

    const rows =
        data.theme_summary;


    const container =
        document.getElementById(
            "themeChart"
        );


    const max =
        Math.max(
            ...rows.map(
                row => row.responses
            )
        );


    container.innerHTML =
        rows.map(
            row => {

                const width =
                    (
                        row.responses
                        / max
                    ) * 100;


                return `
                    <div class="bar-row">

                        <span class="bar-label">
                            ${row.predicted_theme}
                        </span>

                        <div class="bar-track">

                            <div
                                class="bar-fill"
                                style="width:${width}%"
                            ></div>

                        </div>

                        <span class="bar-value">
                            ${row.responses}
                        </span>

                    </div>
                `;

            }
        ).join("");

}


/* ---------------------------------------------------------
   Sentiment
--------------------------------------------------------- */

function renderSentiment(data) {

    const rows =
        data.sentiment_summary;


    const container =
        document.getElementById(
            "sentimentChart"
        );


    const max =
        Math.max(
            ...rows.map(
                row => row.responses
            )
        );


    container.innerHTML =
        rows.map(
            row => {

                const width =
                    (
                        row.responses
                        / max
                    ) * 100;


                return `
                    <div class="bar-row">

                        <span class="bar-label">
                            ${row.sentiment}
                        </span>

                        <div class="bar-track">

                            <div
                                class="bar-fill"
                                style="width:${width}%"
                            ></div>

                        </div>

                        <span class="bar-value">
                            ${row.responses}
                        </span>

                    </div>
                `;

            }
        ).join("");

}


/* ---------------------------------------------------------
   Model comparison
--------------------------------------------------------- */

function renderModels(data) {

    const rows =
        data.model_comparison;


    const container =
        document.getElementById(
            "modelChart"
        );


    container.innerHTML =
        rows.map(
            row => {

                const width =
                    row.accuracy * 100;


                return `
                    <div class="comparison-row">

                        <span class="comparison-label">
                            ${row.approach}
                        </span>

                        <div class="comparison-track">

                            <div
                                class="comparison-fill"
                                style="width:${width}%"
                            ></div>

                        </div>

                        <span class="comparison-value">
                            ${formatPercent(
                                row.accuracy
                            )}
                        </span>

                    </div>
                `;

            }
        ).join("");

}


/* ---------------------------------------------------------
   Formatting
--------------------------------------------------------- */

function formatPercent(value) {

    return (
        value * 100
    ).toFixed(1) + "%";

}


/* ---------------------------------------------------------
   Start
--------------------------------------------------------- */

loadDashboard().catch(
    error => {

        console.error(
            "Dashboard loading failed:",
            error
        );

    }
);