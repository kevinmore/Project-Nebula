#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <Primitives/GameObject.h>

class Camera : public QObject
{
	Q_OBJECT

public:
	Camera(GameObject* followingTarget = 0, QObject *parent = 0);
	~Camera();

	enum ProjectionType
	{
		Orthogonal,
		Perspective
	};

	enum ViewType
	{
		FirstPerson,
		ThirdPerson
	};

	enum CameraTranslationOption
	{
		TranslateViewCenter,
		DontTranslateViewCenter
	};

	vec3 position()   const;
	vec3 upVector()   const;
	vec3 viewCenter() const;
	vec3 viewVector() const;

	ProjectionType projectionType() const;
	void setProjectionType(ProjectionType type);
	
	ViewType viewType() const;
	void setViewType(ViewType type);

	float speed() const;
	float sensitivity() const;
	void setSpeed(float val);
	void setSensitivity(float val);

	float left()   const;
	float right()  const;
	float bottom() const;
	float top()    const;
	void setLeft(const float& left);
	void setRight(const float& right);
	void setBottom(const float& bottom);
	void setTop(const float& top);

	float nearPlane()   const;
	float farPlane()    const;
	float fieldOfView() const;
	float aspectRatio() const;
	void setNearPlane(const float& nearPlane);
	void setFarPlane(const float& nearPlane);
	void setFieldOfView(const float& fieldOfView);
	void setAspectRatio(const float& aspectRatio);
	
	void updateOrthogonalProjection();
	void setOrthographicProjection(float left, float right,	float bottom, float top, float nearPlane, float farPlane);

	void updatePerspectiveProjection();
	void setPerspectiveProjection(float fieldOfView, float aspect, float nearPlane, float farPlane);

	mat4 viewMatrix() const;
	mat4 projectionMatrix() const;
	mat4 viewProjectionMatrix() const;

	quart tiltRotation(const float& angle) const;
	quart rollRotation(const float& angle) const;

	quart panRotation(const float& angle) const;
	quart panRotation(const float& angle, const vec3& axis) const;


	void setPosition(const vec3& position);
	void setUpVector(const vec3& upVector);
	void setViewCenter(const vec3& viewCenter);

	// Translate relative to camera orientation axes
	void translate(const vec3& vLocal, CameraTranslationOption option = TranslateViewCenter);

	// Translate relative to world axes
	void translateWorld(const vec3& vWorld, CameraTranslationOption option = TranslateViewCenter);

	// Translate smoothly with a given duration
	void smoothTransform(const vec3& targetPos, float duration = 0.5);

	void tilt(const float& angle);
	void roll(const float& angle);

	void pan(const float& angle);
	void pan(const float& angle, const vec3& axis);

	void rollAboutViewCenter(const float& angle);
	void tiltAboutViewCenter(const float& angle);
	void panAboutViewCenter(const float& angle);

	void rotate(const quart& q);
	void rotateAboutViewCenter(const quart& q);

	// make the camera follow a game object
	void followTarget(GameObject* target);
	bool isFollowingTarget() { return m_isFollowing; }

	vec3 getViewDirection() { return m_viewDirection; }
	bool isViewCenterFixed() { return m_viewCenterFixed; }

	// camera movement
	inline void setSideSpeed(float vx)     { m_viewDirection.setX(vx); }
	inline void setVerticalSpeed(float vy) { m_viewDirection.setY(vy); }
	inline void setForwardSpeed(float vz)  { m_viewDirection.setZ(vz); }
	inline void setViewCenterFixed(bool b) { m_viewCenterFixed = b; }

	// camera movement rotation
	inline void setPanAngle(float angle)  { m_panAngle  = angle; }
	inline void setTiltAngle(float angle) { m_tiltAngle = angle; }

	void update(const float dt);

public slots:
	void resetCamera();
	void releaseTarget();
	void switchToFirstPersonCamera(bool status);
	void switchToThirdPersonCamera(bool status);

private:
	vec3 m_position;
	vec3 m_upVector;
	vec3 m_viewCenter;
	vec3 m_cameraToCenter;

	ProjectionType m_projectionType;
	ViewType m_viewType;

	float m_nearPlane;
	float m_farPlane;
	float m_fieldOfView;
	float m_aspectRatio;

	float m_left;
	float m_right;
	float m_bottom;
	float m_top;

	float m_speed;
	float m_sensitivity;

	mutable mat4 m_viewMatrix;
	mutable mat4 m_projectionMatrix;
	mutable mat4 m_viewProjectionMatrix;

	mutable bool m_viewMatrixDirty;
	mutable bool m_viewProjectionMatrixDirty;

	// instantiate variables
	vec3 m_viewDirection;
	bool m_viewCenterFixed;
	bool m_isFollowing;
	GameObject* m_followingTarget;

	float m_panAngle;
	float m_tiltAngle;

	const float m_metersPerUnits;
};

