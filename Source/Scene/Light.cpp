#include "Light.h"
#include <Utility/Math.h>
#include <Scene/Scene.h>

using namespace Math;
const float INNER_ANGLE = 30.0f; // 30deg = 0.6981rad = cos(2PI/9) = 0.76604;
const float OUTER_ANGLE = 40.0f; // 40deg = 0.5235rad = cos(1PI/6) = 0.86602;

Light::Light(Scene* scene, GameObject* go) :
	Component(),
	m_scene(scene),
	m_type(PointLight),
	m_position(Math::Vector3::ZERO),
	m_direction(Math::Vector3::NEGATIVE_UNIT_Z),
	m_color(Qt::white),
	m_constantAttenuation(1.0f),
	m_linearAttenuation(0.0f),
	m_quadraticAttenuation(0.0f),
	m_spotFalloff(1.0f),
	m_spotInnerAngle(cosf(static_cast<float>(M_PI)*(INNER_ANGLE/180.0f))),
	m_spotOuterAngle(cosf(static_cast<float>(M_PI)*(OUTER_ANGLE/180.0f))),
	m_intensity(1.0f)
{
	// connect to the scene
	connect(this, SIGNAL(propertiesChanged(Light*)), m_scene, SLOT(onLightChanged(Light*)));
	connect(go, SIGNAL(transformChanged(const vec3&, const vec3&, const vec3&)),
		    this, SLOT(syncTransform(const vec3&, const vec3&, const vec3&)));
}

Light::~Light() 
{
	// remove itself from the light list of the scene
	m_scene->removeLight(this);
}

void Light::setType(LightType type)
{
	m_type = type;

	emit propertiesChanged(this);
}

Light::LightType Light::type() const
{
	return m_type;
}

void Light::setIntensity(float intensity)
{
	m_intensity = intensity;
	emit propertiesChanged(this);
}

float Light::intensity() const
{
	return m_intensity;
}

void Light::setColor(const QColor& color)
{
	if(color.alpha())
		qWarning("The alpha channel of light color should be equal to 0");

	m_color  = color;
	emit propertiesChanged(this);
}

void Light::setColor(float r, float g, float b)
{
	m_color.setRgbF(r, g, b, 0.0);
	emit propertiesChanged(this);
}

const QColor& Light::color() const
{
	return m_color;
}

void Light::setPosition(const vec3& position)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use position property");

	m_position = position;

	emit propertiesChanged(this);
}

void Light::setPosition(float x, float y, float z)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use position property");

	m_position.setX(x);
	m_position.setY(y);
	m_position.setZ(z);

	emit propertiesChanged(this);
}

const vec3& Light::position() const
{
	return m_position;
}

void Light::setDirection(const vec3& direction)
{
	if(m_type == PointLight)
		qWarning("Point lights not use direction property");

	m_direction = direction;

	emit propertiesChanged(this);
}

void Light::setDirection(float x, float y, float z)
{
	if(m_type == PointLight)
		qWarning("Point lights not use direction property");

	m_direction.setX(x);
	m_direction.setY(y);
	m_direction.setZ(z);

	emit propertiesChanged(this);
}

const vec3 Light::direction() const
{
	return m_direction.normalized();
}

void Light::setAttenuation(float constantFactor,
	float linearFactor,
	float quadraticFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use attenuation property");

	m_constantAttenuation  = constantFactor;
	m_linearAttenuation    = linearFactor;
	m_quadraticAttenuation = quadraticFactor;

	emit propertiesChanged(this);
}

void Light::setConstantAttenuation(float constantFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use constant attenuation property");

	m_constantAttenuation = constantFactor;

	emit propertiesChanged(this);
}

void Light::setLinearAttenuation(float linearFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use linear attenuation property");

	m_linearAttenuation = linearFactor;

	emit propertiesChanged(this);
}

void Light::setQuadraticAttenuation(float quadraticFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use quadratic attenuation property");

	m_quadraticAttenuation = quadraticFactor;

	emit propertiesChanged(this);
}

float Light::constantAttenuation() const
{
	return m_constantAttenuation;
}

float Light::linearAttenuation() const
{
	return m_linearAttenuation;
}

float Light::quadraticAttenuation() const
{
	return m_quadraticAttenuation;
}

void Light::setSpotFalloff(float falloff)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set falloff property");

	m_spotFalloff = falloff;

	emit propertiesChanged(this);
}

void Light::setSpotInnerAngle(float innerAngle)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set inner angle value");

	m_spotInnerAngle = cosf(static_cast<float>(M_PI)*(innerAngle/180.0f));

	emit propertiesChanged(this);
}

void Light::setSpotOuterAngle(float outerAngle)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set outer angle value");

	m_spotOuterAngle = cosf(static_cast<float>(M_PI)*(outerAngle/180.0f));

	emit propertiesChanged(this);
}

float Light::spotFallOff() const
{
	return m_spotFalloff;
}

float Light::spotInnerAngle() const
{
	return m_spotInnerAngle;
}

float Light::spotOuterAngle() const
{
	return m_spotOuterAngle;
}

void Light::syncTransform( const vec3& pos, const vec3& rot, const vec3& scale )
{
	// synchronize the position and rotation from the game object
	// that this light attached to
	m_position = pos;
	m_direction = rot;

	emit propertiesChanged(this);
}
