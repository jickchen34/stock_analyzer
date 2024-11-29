// 加载数据
fetch("data/data.json")
  .then((response) => response.json())
  .then((data) => {
    // 处理时间数据
    const times = data.map((item) => item.log_time);
    const timeMs = times.map((time) => {
      const [date, timeWithMs] = time.split(" ");
      return timeWithMs;
    });

    // 初始化图表
    const chartDom = document.getElementById("timeDistChart");
    const myChart = echarts.init(chartDom);

    // 配置项
    const option = {
      title: {
        text: "Trading Time Distribution",
      },
      tooltip: {
        trigger: "axis",
        formatter: function (params) {
          return params[0].value;
        },
      },
      xAxis: {
        type: "category",
        data: timeMs,
        axisLabel: {
          rotate: 45,
          interval: 100,
        },
      },
      yAxis: {
        type: "value",
        show: true,
      },
      dataZoom: [
        {
          type: "slider",
          show: true,
          xAxisIndex: [0],
          start: 0,
          end: 100,
        },
        {
          type: "inside",
          xAxisIndex: [0],
          start: 0,
          end: 100,
        },
      ],
      series: [
        {
          name: "Time",
          type: "scatter",
          data: timeMs.map((_, index) => index),
          symbolSize: 5,
          itemStyle: {
            color: "#4169E1",
          },
        },
      ],
    };

    // 使用配置项显示图表
    myChart.setOption(option);
  });
