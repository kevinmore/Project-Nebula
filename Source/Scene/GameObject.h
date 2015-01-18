#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <QSharedPointer>
#include <QElapsedTimer>
using namespace Math;

class Component;
class AbstractModel;
typedef QSharedPointer<Component> ComponentPtr;
typedef QSharedPointer<AbstractModel> ModelPtr;

class GameObject : public QObject
{
	Q_OBJECT
	Q_PROPERTY(bool moving READ isMoving WRITE setMoving)
	Q_PROPERTY(vec3 localSpeed READ localSpeed WRITE setSpeed)

public:
	GameObject(QObject* parent = 0);
	~GameObject();

	bool isMoving() const;
	void setMoving(bool status);

	void setPosition(const QVector3D& positionVector);
	void setPosition(double x, double y, double z);
	void translateInWorld(const QVector3D& delta);

	void setRotation(const QVector3D& rotationVector);
	void setRotation(double x, double y, double z);
	void rotateInWorld(const QQuaternion& delta);

	void setScale(const QVector3D& scale);
	void setScale(double x, double y, double z);
	void setScale(double scaleFactor);

	void setSpeed(const QVector3D& speed);
	void setSpeed(double x, double y, double z);

	QVector3D globalSpeed() const;
	QVector3D predictedPosition() const;

	/////////////////////////////inline section///////////////////////////////////
	inline QVector3D position() const { return m_position; }
	inline QVector3D rotation() const { return m_rotation; }
	inline QVector3D scale() const { return m_scale; }
	inline QVector3D localSpeed() const { return m_speed; }

	inline const QMatrix4x4& modelMatrix()
	{
		if(m_modelMatrixDirty)
		{
			m_modelMatrix.setToIdentity();

			m_modelMatrix.translate(m_position);
			m_modelMatrix.rotate(m_rotation.x(), Vector3D::UNIT_X);
			m_modelMatrix.rotate(m_rotation.y(), Vector3D::UNIT_Y);
			m_modelMatrix.rotate(m_rotation.z(), Vector3D::UNIT_Z);
			m_modelMatrix.scale(m_scale);

			m_modelMatrixDirty = false;
		}

		return m_modelMatrix;
	}

	inline int renderOrder() const { return m_renderOrder; }
	/////////////////////////////inline section///////////////////////////////////

	void attachModel(ModelPtr pModel);
	ModelPtr getModel();

	void attachComponent(ComponentPtr pComponent);
	QVector<ComponentPtr> getComponents();
	ComponentPtr getComponent(const QString& name);

	enum MovingBehaviour
	{
		CONSECUTIVE,
		DISCRETE
	};

	void setMovingBehaviour(MovingBehaviour type);
	MovingBehaviour movingBehaviour() const;

signals:
	void synchronized();

public slots:
	void translateX(double x);
	void translateY(double y);
	void translateZ(double z);

	void rotateX(double x);
	void rotateY(double y);
	void rotateZ(double z);

	void scaleX(double x);
	void scaleY(double y);
	void scaleZ(double z);

	void translateInWorld(const QString& paramString);
	void rotateInWorld(const QString& paramString);
	void rotateInWorldAxisAndAngle(const QString& paramString);
	void setLocalSpeed(const QString& paramString);
	void resetSpeed();

	void calculateSpeed();

	void reset();

private:
	QVector3D m_position, m_prevPosition;
	QVector3D m_rotation;
	QVector3D m_scale;
	QVector3D m_speed;

	QMatrix4x4 m_modelMatrix;

	bool m_modelMatrixDirty;

	MovingBehaviour m_movingBehaviour;
	
	float m_time;
	QElapsedTimer m_lifeTimer;
	bool m_isMoving;
	ModelPtr m_model;
	QVector<ComponentPtr> m_components;
	int m_renderOrder;
};

