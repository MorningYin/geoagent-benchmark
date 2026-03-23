# GeoAgentBench App 架构设计

> 本文档定义 5 个地理领域 App 的数据模型和工具接口。
> 设计原则：从 Google Maps API 能提供什么数据出发，区分静态基底和动态覆盖层，推导出 App 结构和 Agent 工具。

---

## 一、数据分层：基底 vs 覆盖层

### 从 Google Maps API 能拿到什么

```
Places API (Nearby Search + Details):
  id, displayName, formattedAddress, location{lat,lng}
  types[], primaryType, rating, userRatingCount
  regularOpeningHours{periods[], weekdayDescriptions[]}
  currentOpeningHours{openNow}
  websiteUri, nationalPhoneNumber

Geocoding API:
  formatted_address, address_components[], place_id, types

Routes API (Directions):
  routes[].legs[].distance{value,text}, duration{value,text}
  steps[]{travel_mode, polyline, distance, duration, instructions}

Street View:
  JPEG 图片字节（无结构化元数据）
```

### 什么是静态的（不随场景变化）

```
✓ 地点的存在性           — 一家餐厅存在就是存在
✓ 地点的位置             — lat/lng 不会变
✓ 地点的名称、地址、类型  — 结构化属性
✓ 地点的常规营业时间      — regularOpeningHours（周一到周日的固定时间表）
✓ 地点的评分和评价数量    — 快照值，不随场景变
✓ 地点的联系方式          — 电话、网站
✓ 两点之间的物理距离      — 地球表面距离不变
✓ 步行/驾车基准时间       — 不含实时路况的基准值
✓ 公交线路和站点         — 基础设施（线路存在、站点位置）
✓ 地理编码映射           — 地址 ↔ 坐标
```

→ 这些构成 **区域基底（Area Base）**，一次抓取，所有场景共享。

### 分层判断标准：实体 vs 状态

**实体**（被人创造出来、持久存在）→ 基底（共享）
**状态**（描述当前世界条件）→ 覆盖层（独有）

```
实体（需要基底）:
  ✓ 地点（餐厅存在就是存在）
  ✓ 评论（被写出来了，不会消失）
  ✓ 社交帖子（被发布了，持续存在）
  ✓ 酒店（住宿设施持续存在）
  ✓ 票价（景点定好的标准价格）
  ✓ 常驻演出（长期运营的活动）
  ✓ 公交线路（基础设施）

状态（纯覆盖层）:
  ✗ 天气（纯粹的当前条件）
  ✗ 交通拥堵（实时状态）
  ✗ 公交延误/停运（临时状态）
  ✗ 临时关闭（临时状态）
  ✗ 预订可用性（当日状态）
  ✗ 排队时间（实时状态）
  ✗ 临时活动（短期存在）
```

### 完整分层表

| App | 基底（实体，共享，一次性构建） | 覆盖层（状态，独有，每场景生成） | 基底来源 |
|-----|--------------------------|-------------------------------|---------|
| MapApp | 500-2000 POI + 路线 + 公交线路 | 临时关闭、交通状况、公交延误 | Google Maps API |
| LocalLifeApp | 50 商户 × 5-8 条评论 ≈ 300 条 | 场景特需的关键评论 + 当日订位状态 | LLM（引用 MapApp POI） |
| SocialApp | 100-150 条区域帖子 | 朋友帖子 + 当前趋势 | LLM（引用 MapApp 地标） |
| BookingApp | 15-20 酒店 + 10-15 票价 + 5-10 常驻活动 | 具体日期可用性 + 排队 + 临时活动 | LLM（引用 MapApp POI） |
| WeatherApp | (无) | 当前天气 + 预报 | LLM |

### 基底的作用

基底提供**噪声和密度**，不是提供场景答案：
```
没有基底: search_reviews("安静") → 返回 2 条 → 必然是答案 → 太容易
有基底:   search_reviews("安静") → 返回 8 条（基底 5 + 覆盖 3）
          → "环境安静适合读书"（有用）
          → "安静是因为没人来"（有用但相反）
          → "老板很安静"（噪声）
          → Agent 必须理解语义 → 真正的推理
```

### 基底的构建流程

```
Step 1: MapApp 基底（Google Maps API）← 地基
  nearby_search + place_details → 500-2000 POI
  Routes API → 关键路线缓存
  成本: ~$15/区域

Step 2: 从 MapApp 提取关键实体
  餐厅/咖啡馆 → 给 LocalLifeApp 生成评论
  景点/博物馆 → 给 BookingApp 生成票价
  地标/网红点 → 给 SocialApp 生成帖子
  酒店/民宿   → 给 BookingApp 生成房源

Step 3: LLM 生成关联基底（引用 MapApp 真实 place_id 和地名）
  3a: LocalLifeApp — 为 50 家商户生成 300 条评论 (~$1.5)
  3b: SocialApp — 为区域生成 120 条帖子 (~$1.0)
  3c: BookingApp — 酒店+票价+常驻活动 (~$0.5)

每区域总计: ~$18 (API $15 + LLM $3)
30 区域: ~$540
```

### 交叉引用约束

所有非 MapApp 基底必须引用 MapApp 的真实数据：
```
LocalLifeApp.reviews[i].place_id  ∈  MapApp.places[].id
SocialApp.posts[i].location_name  ∈  MapApp.places[].name（或区域地标名）
BookingApp.tickets[i].attraction_id  ∈  MapApp.places[].id (type=tourist_attraction)
BookingApp.hotels[i].location  在  MapApp 的区域范围内
```

---

## 二、数据模型

### 基础类型（跨 App 共用）

```python
@dataclass
class Location:
    latitude: float
    longitude: float

@dataclass
class Place:
    """区域基底中的地点。字段直接来自 Google Maps API。"""
    id: str                              # Google place_id, e.g. "ChIJ..."
    name: str                            # displayName.text
    address: str                         # formattedAddress
    location: Location
    types: list[str]                     # e.g. ["restaurant", "food"]
    primary_type: str                    # e.g. "restaurant"
    rating: float | None                 # e.g. 4.4
    rating_count: int                    # e.g. 7448
    phone: str | None                    # nationalPhoneNumber
    website: str | None                  # websiteUri
    regular_hours: dict | None           # regularOpeningHours (周一到周日)
    # 注意：currentOpeningHours 不在基底里，由覆盖层动态决定
```

### MapApp 数据模型

```python
@dataclass
class Route:
    origin: Location
    destination: Location
    mode: str                            # "walking" | "driving" | "transit"
    distance_m: int                      # 距离（米）
    duration_min: float                  # 基准时间（分钟，不含实时路况）
    steps: list[RouteStep] | None        # 详细步骤（可选）

@dataclass
class RouteStep:
    instruction: str                     # "沿南京路向东步行200米"
    distance_m: int
    duration_min: float
    travel_mode: str

@dataclass
class TransitLine:
    line_id: str                         # "line_metro_2"
    name: str                            # "地铁2号线"
    type: str                            # "metro" | "bus" | "tram"
    stops: list[TransitStop]

@dataclass
class TransitStop:
    stop_id: str
    name: str
    location: Location
    lines: list[str]                     # 经过该站的线路 ID 列表

@dataclass
class TransitAlert:
    """覆盖层：公交动态变更"""
    line_id: str
    alert_type: str                      # "delay" | "suspension" | "reroute"
    message: str                         # "2号线因故障暂停运营"
    affected_stops: list[str]            # 受影响站点 ID

@dataclass
class TrafficCondition:
    """覆盖层：交通状况"""
    region: str                          # 描述区域
    congestion_level: str                # "low" | "moderate" | "heavy" | "severe"
    delay_factor: float                  # 1.0=正常, 1.5=慢50%, 2.0=慢一倍

@dataclass
class TemporaryClosure:
    """覆盖层：临时关闭"""
    place_id: str
    reason: str                          # "装修" | "临时休息" | "包场"
    closed_until: str | None             # "2026-03-25" 或 None=不确定
```

### LocalLifeApp 数据模型

```python
@dataclass
class Review:
    review_id: str
    place_id: str                        # 关联到 MapApp 的 Place.id
    author: str
    rating: int                          # 1-5
    text: str                            # 评论正文
    date: str                            # "2026-01-15"
    photos: list[str]                    # 图片 URL/ID 列表
    helpful_count: int

@dataclass
class Reservation:
    reservation_id: str
    place_id: str
    party_size: int
    datetime: str                        # "2026-03-24 19:00"
    name: str
    status: str                          # "confirmed" | "cancelled"

@dataclass
class ReservationSlot:
    """某商家的可预订状态"""
    place_id: str
    available: bool
    time_slots: list[str]                # ["18:00", "18:30", "19:00", ...]
    max_party_size: int
```

### SocialApp 数据模型

```python
@dataclass
class Post:
    post_id: str
    author: str
    content: str                         # 帖子正文
    images: list[str]                    # 图片描述或 ID
    location: Location | None            # 定位标签（可能为空）
    location_name: str | None            # "西湖断桥"
    tags: list[str]                      # ["打卡", "美食", "杭州"]
    likes: int
    date: str
```

### WeatherApp 数据模型

```python
@dataclass
class WeatherData:
    location: Location
    time: str                            # ISO datetime
    temperature_c: float
    condition: str                       # "clear"|"cloudy"|"rain"|"snow"|"fog"|"storm"
    humidity_pct: int
    wind_speed_kmh: float
    wind_direction: str                  # "N"|"NE"|"E"|...
    uv_index: int
    precipitation_mm: float
    description: str                     # "多云转阴，午后有阵雨"

@dataclass
class SunTimes:
    sunrise: str                         # "06:23"
    sunset: str                          # "18:45"
```

### BookingApp 数据模型

```python
@dataclass
class TicketInfo:
    attraction_id: str                   # 关联到 MapApp 的 Place.id
    ticket_types: dict[str, float]       # {"adult": 60.0, "child": 30.0, "student": 40.0}
    booking_required: bool
    available_dates: list[str]           # ["2026-03-24", "2026-03-25"]
    sold_out_dates: list[str]            # ["2026-03-23"]
    time_slots: list[str] | None         # ["09:00-12:00", "13:00-17:00"]

@dataclass
class WaitTimeInfo:
    attraction_id: str
    current_wait_minutes: int
    estimated_entry_time: str            # "14:30"

@dataclass
class Hotel:
    hotel_id: str
    name: str
    location: Location
    address: str
    star_rating: int                     # 1-5
    price_range: dict[str, float]        # {"standard": 400, "deluxe": 800}
    amenities: list[str]                 # ["wifi", "parking", "pool"]
    available: bool

@dataclass
class EventInfo:
    event_id: str
    name: str                            # "宋城千古情"
    venue: str                           # "杭州宋城"
    venue_location: Location
    date: str
    time: str                            # "14:00-15:30"
    price: float
    tickets_remaining: int

@dataclass
class Booking:
    booking_id: str
    type: str                            # "ticket" | "hotel" | "event"
    item_id: str                         # attraction_id / hotel_id / event_id
    details: dict                        # 预订详情
    status: str                          # "confirmed" | "cancelled" | "modified"
```

---

## 三、App 工具接口设计

### 设计原则（从 ARE 学到的模式）

```
1. 分页: 返回列表的方法都带 offset/limit，返回 {results, range, total}
2. 错误: 统一用 ValueError，消息要清晰可恢复
3. 搜索: 大小写不敏感的子串匹配（和 ARE 一致）
4. 时间: 输入用 "YYYY-MM-DD HH:MM:SS" 字符串，内部存 float timestamp
5. ID: 用 place_id（来自 Google）或 uuid4().hex（自生成）
6. 读方法: 返回数据对象或分页结果
7. 写方法: 返回新建 ID 或确认消息
8. 确定性: 同样的输入 + 同样的 state → 同样的输出（无随机性）
```

### MapApp — 地图服务（高德/Google Maps）

```python
class MapApp(App):
    """
    地图服务，提供地点搜索、地理编码、路线规划、公交查询、交通状况。

    State 结构:
      base: {places[], routes_cache{}, transit_lines[], transit_stops[]}  ← 区域基底
      overlay: {closures[], traffic[], transit_alerts[]}                  ← 场景覆盖
      saved_places: {}                                                   ← Agent 写入
      saved_routes: []                                                   ← Agent 写入
    """

    # ═══ READ: 地点搜索 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_places(self, query: str, latitude: float, longitude: float,
                      radius_m: int = 1000, type_filter: str | None = None,
                      offset: int = 0, limit: int = 10) -> dict:
        """Search for places near a location.

        Args:
            query: Search keyword (e.g. "餐厅", "pharmacy")
            latitude, longitude: Center point
            radius_m: Search radius in meters
            type_filter: Filter by primary_type (e.g. "restaurant")
            offset, limit: Pagination

        Returns: {results: list[Place], range: (start, end), total: int}
        """
        # 内部逻辑: 从 base.places 中筛选距离 ≤ radius_m 的，
        # 且 (query 匹配 name/types/address) 且 (type_filter 匹配 primary_type)
        # 排除 overlay.closures 中的临时关闭地点
        # 按距离排序

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_place_details(self, place_id: str) -> dict:
        """Get detailed information about a specific place.

        Returns: Place object with all fields + is_currently_open status
        """
        # 内部逻辑: 从 base.places 查找
        # 结合 overlay.closures 判断当前是否开放
        # 结合场景时间 + regular_hours 判断常规营业状态

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def geocode(self, address: str) -> dict:
        """Convert address text to coordinates.

        Returns: {latitude, longitude, formatted_address, place_id}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def reverse_geocode(self, latitude: float, longitude: float) -> dict:
        """Convert coordinates to address.

        Returns: {formatted_address, address_components[], place_id}
        """

    # ═══ READ: 路线规划 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_directions(self, origin_lat: float, origin_lng: float,
                       dest_lat: float, dest_lng: float,
                       mode: str = "walking") -> dict:
        """Get directions between two points.

        Args:
            mode: "walking" | "driving" | "transit" | "bicycling"

        Returns: Route object {distance_m, duration_min, steps[]}
                 driving 模式下 duration_min 会受 overlay.traffic 影响
        """
        # 内部逻辑: 查 routes_cache，如果有缓存直接返回
        # driving 模式: base duration × overlay.traffic.delay_factor
        # transit 模式: 检查 overlay.transit_alerts 看有没有停运

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_distance_matrix(self, origins: list[dict], destinations: list[dict],
                            mode: str = "walking") -> dict:
        """Get distance/duration matrix between multiple points.

        Args:
            origins: [{latitude, longitude}, ...]
            destinations: [{latitude, longitude}, ...]

        Returns: {rows: [{elements: [{distance_m, duration_min}, ...]}]}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_travel_time(self, origin_lat: float, origin_lng: float,
                        dest_lat: float, dest_lng: float,
                        mode: str = "walking") -> dict:
        """Quick travel time query (convenience wrapper).

        Returns: {distance_m, duration_min, mode}
        """

    # ═══ READ: 公交查询 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_nearby_stations(self, latitude: float, longitude: float,
                            radius_m: int = 500) -> list:
        """Find transit stations near a point.

        Returns: list of {stop_id, name, location, lines[], distance_m}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_transit_route(self, origin_lat: float, origin_lng: float,
                          dest_lat: float, dest_lng: float) -> dict:
        """Plan a public transit route.

        Returns: {segments: [{line, board_at, alight_at, duration_min}, ...],
                  total_duration_min, transfers}
                  会考虑 overlay.transit_alerts 的停运信息
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_transit_alerts(self) -> list:
        """Get current transit service alerts.

        Returns: list of TransitAlert
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_traffic_conditions(self, latitude: float, longitude: float,
                               radius_m: int = 2000) -> dict:
        """Get current traffic conditions in an area.

        Returns: {congestion_level, delay_factor, description}
        """

    # ═══ WRITE ═══

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def save_place(self, place_id: str, list_name: str = "favorites") -> str:
        """Save a place to a named list.

        Returns: confirmation message
        """

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def save_route(self, origin_lat: float, origin_lng: float,
                   dest_lat: float, dest_lng: float,
                   mode: str, name: str) -> str:
        """Save a route for later reference.

        Returns: route_id
        """

    # ═══ ENV: 场景初始化 ═══

    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def set_temporary_closure(self, place_id: str, reason: str,
                              closed_until: str | None = None) -> str:
        """Mark a place as temporarily closed."""

    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def set_traffic_condition(self, region: str, congestion_level: str,
                              delay_factor: float) -> str:
        """Set traffic conditions for a region."""

    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def add_transit_alert(self, line_id: str, alert_type: str,
                          message: str, affected_stops: list[str]) -> str:
        """Add a transit service alert."""
```

### LocalLifeApp — 本地生活（大众点评/Yelp）

```python
class LocalLifeApp(App):
    """
    本地生活服务，提供商家评论、搜索评论、餐厅订位。

    State 结构:
      reviews: {place_id: [Review, ...]}
      reservation_slots: {place_id: ReservationSlot}
      reservations: {reservation_id: Reservation}       ← Agent 写入
    """

    # ═══ READ ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_reviews(self, place_id: str, sort_by: str = "date",
                    offset: int = 0, limit: int = 5) -> dict:
        """Get reviews for a place.

        Args:
            sort_by: "date" | "rating_high" | "rating_low" | "helpful"

        Returns: {reviews: list[Review], range, total}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_reviews(self, place_id: str, keyword: str,
                       offset: int = 0, limit: int = 5) -> dict:
        """Search reviews containing specific keywords.

        Args:
            keyword: e.g. "安静", "素食", "包间"

        Returns: {reviews: list[Review], range, total}
        """
        # 内部逻辑: 大小写不敏感子串匹配 review.text

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def check_reservation(self, place_id: str, party_size: int,
                          datetime: str) -> dict:
        """Check if a reservation is available.

        Returns: {available: bool, alternative_times: list[str]}
        """

    # ═══ WRITE ═══

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def make_reservation(self, place_id: str, party_size: int,
                         datetime: str, name: str) -> str:
        """Make a restaurant reservation.

        Returns: reservation_id
        """

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> str:
        """Cancel a reservation.

        Returns: confirmation message
        """

    # ═══ ENV ═══

    @env_tool()
    def add_reviews(self, place_id: str, reviews: list[dict]) -> str:
        """Load reviews for a place (scenario setup)."""

    @env_tool()
    def set_reservation_availability(self, place_id: str,
                                      available: bool,
                                      time_slots: list[str],
                                      max_party_size: int) -> str:
        """Set reservation availability for a place."""
```

### SocialApp — 社交媒体（小红书/Instagram）

```python
class SocialApp(App):
    """
    社交媒体，提供旅行帖子搜索、趋势内容、用户帖子。

    State 结构:
      posts: [Post, ...]
      trending: {category: [Post, ...]}
      user_posts: {user_id: [Post, ...]}
    """

    # ═══ READ ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_posts(self, query: str,
                     latitude: float | None = None,
                     longitude: float | None = None,
                     radius_m: int = 5000,
                     offset: int = 0, limit: int = 10) -> dict:
        """Search posts by keyword and optional location.

        Returns: {posts: list[Post], range, total}
        """
        # 内部逻辑: query 匹配 content/tags，位置在 radius 内

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_trending(self, latitude: float, longitude: float,
                     category: str | None = None) -> list:
        """Get trending posts near a location.

        Args:
            category: "food" | "travel" | "nightlife" | None (all)

        Returns: list[Post] (sorted by likes, top 10)
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_user_posts(self, user_id: str,
                       offset: int = 0, limit: int = 10) -> dict:
        """Get posts by a specific user.

        Returns: {posts: list[Post], range, total}
        """

    # ═══ WRITE ═══

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def share_location(self, content: str,
                       latitude: float, longitude: float,
                       location_name: str | None = None,
                       tags: list[str] | None = None) -> str:
        """Share current location with a post.

        Returns: post_id
        """

    # ═══ ENV ═══

    @env_tool()
    def add_posts(self, posts: list[dict]) -> str:
        """Load posts (scenario setup)."""
```

### WeatherApp — 天气预报

```python
class WeatherApp(App):
    """
    天气服务，提供当前天气、预报、日出日落。

    State 结构:
      current_weather: WeatherData
      forecast: [WeatherData, ...]       # 未来 24-72 小时，每小时一条
      sun_times: {date: SunTimes}
    """

    # ═══ READ ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_current_weather(self, latitude: float,
                            longitude: float) -> dict:
        """Get current weather conditions.

        Returns: WeatherData
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_forecast(self, latitude: float, longitude: float,
                     hours: int = 24) -> list:
        """Get weather forecast.

        Args:
            hours: How many hours ahead (max 72)

        Returns: list[WeatherData]
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_sunrise_sunset(self, latitude: float, longitude: float,
                           date: str) -> dict:
        """Get sunrise and sunset times for a date.

        Returns: {sunrise: "06:23", sunset: "18:45"}
        """

    # ═══ WRITE ═══

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def set_weather_alert(self, condition: str, message: str) -> str:
        """Set a weather alert reminder.

        Returns: alert_id
        """

    # ═══ ENV ═══

    @env_tool()
    def set_weather(self, current: dict, forecast: list[dict],
                    sun_times: dict | None = None) -> str:
        """Load weather data (scenario setup)."""
```

### BookingApp — 预订服务（携程/Booking）

```python
class BookingApp(App):
    """
    预订服务，提供景点门票、酒店预订、活动查询。

    State 结构:
      tickets: {attraction_id: TicketInfo}
      wait_times: {attraction_id: WaitTimeInfo}
      hotels: {hotel_id: Hotel}
      events: [EventInfo, ...]
      bookings: {booking_id: Booking}     ← Agent 写入
    """

    # ═══ READ: 门票 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_ticket_price(self, attraction_id: str) -> dict:
        """Get ticket pricing for an attraction.

        Returns: TicketInfo
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def check_attraction_availability(self, attraction_id: str,
                                      date: str) -> dict:
        """Check if tickets are available for a specific date.

        Returns: {available: bool, time_slots: list[str], sold_out: bool}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_wait_time(self, attraction_id: str) -> dict:
        """Get current wait time at an attraction.

        Returns: WaitTimeInfo
        """

    # ═══ READ: 酒店 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_hotels(self, latitude: float, longitude: float,
                      radius_m: int = 2000,
                      check_in: str | None = None,
                      check_out: str | None = None,
                      guests: int = 1,
                      offset: int = 0, limit: int = 10) -> dict:
        """Search for hotels near a location.

        Returns: {hotels: list[Hotel], range, total}
        """

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def check_room_availability(self, hotel_id: str,
                                check_in: str, check_out: str) -> dict:
        """Check room availability at a specific hotel.

        Returns: {available: bool, room_types: list, price_per_night: float}
        """

    # ═══ READ: 活动 ═══

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_events(self, latitude: float, longitude: float,
                      radius_m: int = 5000,
                      date: str | None = None,
                      category: str | None = None,
                      offset: int = 0, limit: int = 10) -> dict:
        """Search for events and shows.

        Returns: {events: list[EventInfo], range, total}
        """

    # ═══ WRITE ═══

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def book_ticket(self, attraction_id: str, ticket_type: str,
                    date: str, count: int = 1) -> str:
        """Book tickets for an attraction.

        Returns: booking_id
        Raises: ValueError if sold out or invalid date
        """

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def book_hotel(self, hotel_id: str, check_in: str,
                   check_out: str, guests: int,
                   room_type: str = "standard") -> str:
        """Book a hotel room.

        Returns: booking_id
        """

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def cancel_booking(self, booking_id: str) -> str:
        """Cancel a booking.

        Returns: confirmation message with refund info
        """

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def modify_booking(self, booking_id: str,
                       new_date: str | None = None,
                       new_count: int | None = None) -> str:
        """Modify an existing booking.

        Returns: confirmation message
        """

    # ═══ ENV ═══

    @env_tool()
    def add_ticket_info(self, attraction_id: str, ticket_types: dict,
                        booking_required: bool,
                        available_dates: list[str],
                        sold_out_dates: list[str] | None = None) -> str:
        """Set ticket info for an attraction."""

    @env_tool()
    def add_hotel(self, hotel: dict) -> str:
        """Add a hotel to the system."""

    @env_tool()
    def add_event(self, event: dict) -> str:
        """Add an event/show."""

    @env_tool()
    def set_wait_time(self, attraction_id: str,
                      wait_minutes: int) -> str:
        """Set current wait time."""
```

---

## 四、接口统计

| App | READ 方法 | WRITE 方法 | ENV 方法 | 数据模型 |
|-----|----------|-----------|---------|---------|
| MapApp | 11 | 2 | 3 | Place, Route, TransitLine, TransitStop, TransitAlert, TrafficCondition, TemporaryClosure |
| LocalLifeApp | 3 | 2 | 2 | Review, Reservation, ReservationSlot |
| SocialApp | 3 | 1 | 1 | Post |
| WeatherApp | 3 | 1 | 1 | WeatherData, SunTimes |
| BookingApp | 6 | 4 | 4 | TicketInfo, WaitTimeInfo, Hotel, EventInfo, Booking |
| **总计** | **26** | **10** | **11** | **17 个数据类** |

Agent 可用工具: 26 READ + 10 WRITE = **36 个 @app_tool**
场景构建工具: 11 个 @env_tool

---

## 五、场景覆盖的验证

用"4人约饭"场景验证接口完整性:

```
1. MapApp.reverse_geocode(lat, lng)              ✓ 定位
2. MapApp.search_places("餐厅", lat, lng, 2000)  ✓ 搜索候选
3. LocalLifeApp.search_reviews(place_id, "素食") ✓ 筛选约束
4. LocalLifeApp.search_reviews(place_id, "包间") ✓ 筛选约束
5. MapApp.get_distance_matrix(各人位置, 餐厅)    ✓ 位置优化
6. LocalLifeApp.check_reservation(place_id, 4, "19:00") ✓ 检查可订
7. LocalLifeApp.make_reservation(place_id, 4, "19:00", "张三") ✓ 订位
8. MessagingApp.send_message(...)                 ✓ 通知朋友 (ARE 已有)
```

用"台风改行程"场景验证:

```
Turn 1:
1. WeatherApp.get_forecast(lat, lng, 48)          ✓ 查预报
2. CalendarApp.get_events(...)                    ✓ 查行程 (ARE)

Turn 2 (台风来了):
3. WeatherApp.get_current_weather(lat, lng)       ✓ 确认天气
4. BookingApp.cancel_booking(ship_ticket_id)      ✓ 退船票
5. BookingApp.modify_booking(hotel_id, new_date)  ✓ 改酒店
6. MapApp.search_places("博物馆", lat, lng)       ✓ 找室内替代
7. BookingApp.check_attraction_availability("博物馆", date) ✓ 查预约
8. BookingApp.book_ticket("博物馆", "adult", date, 2) ✓ 订票
9. MapApp.get_directions(hotel, museum, "transit") ✓ 新路线
10. CalendarApp.update_event(...)                  ✓ 改行程 (ARE)
11. MessagingApp.send_message(...)                 ✓ 通知同伴 (ARE)
```

两个场景都完整覆盖，无缺失接口。
