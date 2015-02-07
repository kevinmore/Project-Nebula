#pragma once
#include <Utility/EngineCommon.h>
#include <Primitives/Component.h>

class Scene;
class Light : public Component
{

	Q_OBJECT

public:
	enum LightType
	{
		PointLight,
		DirectionalLight,
		SpotLight
	};

	Light(Scene* scene, GameObject* go);
	~Light();

	virtual QString className() { return "Light"; }
	virtual void render(const float currentTime) {/*do noting*/}

	void setType(LightType type);
	LightType type() const;

	void setIntensity(float intensity);
	float intensity() const;

	void setColor(const QColor& color);
	void setColor(float r, float g, float b);

	const QColor& color() const;

	void setDirection(const vec3& direction);
	void setDirection(float x, float y, float z);

	const vec3 direction() const;

	void setPosition(const vec3& position);
	void setPosition(float x, float y, float z);

	const vec3& position() const;

	void setAttenuation(float constantFactor, float linearFactor, float quadraticFactor);

	void setConstantAttenuation(float constantFactor);
	void setLinearAttenuation(float linearFactor);
	void setQuadraticAttenuation(float quadraticFactor);

	float constantAttenuation() const;
	float linearAttenuation() const;
	float quadraticAttenuation() const;

	void setSpotFalloff(float falloff);
	void setSpotInnerAngle(float innerAngle);
	void setSpotOuterAngle(float outerAngle);

	float spotFallOff() const;
	float spotInnerAngle() const;
	float spotOuterAngle() const;

signals:
	void propertiesChanged(Light* l);

private slots:
	void syncTransform(const vec3& pos, const vec3& rot, const vec3& scale);

private:
	Scene* m_scene;

	LightType m_type;

	vec3 m_position;
	vec3 m_direction;

	QColor m_color;

	float m_constantAttenuation;
	float m_linearAttenuation;
	float m_quadraticAttenuation;
	float m_spotFalloff;
	float m_spotInnerAngle;
	float m_spotOuterAngle;
	float m_intensity;
};

typedef QSharedPointer<Light> LightPtr;
