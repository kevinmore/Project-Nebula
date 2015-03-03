/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/Action/hkpAction.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Physics2012/Vehicle/hkpVehicleInstance.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Vehicle/hkpVehicleViewer.h>

int			hkpVehicleViewer::m_tag				= 0;
const int	hkpVehicleViewer::s_numWheelSegments	= 12;

// A Little macro to convert degrees to radians
#define DEGREES2RADIANS(deg) ((deg)*(0.0174532925f))

void HK_CALL hkpVehicleViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpVehicleViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpVehicleViewer(contexts);
}

hkpVehicleViewer::hkpVehicleViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase(contexts)
{	
}

void hkpVehicleViewer::init()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			worldAddedCallback( m_context->getWorld(i));
		}
	}
}

hkpVehicleViewer::~hkpVehicleViewer()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			worldRemovedCallback( m_context->getWorld(i));
		}
	}
}

void hkpVehicleViewer::worldAddedCallback( hkpWorld* w)
{
	w->markForWrite();
	w->addActionListener(this);
	w->addWorldPostSimulationListener(this);

	// This class is an action listener, and thus will be notified when actions
	// are added to	the world. However, since this class can be created by the
	// user in the VDB, there may already be actions added to the world when
	// this class is created. This loops finds any existing actions and passes
	// them to the action added callback.
	hkpPhysicsSystem* system = w->getWorldAsOneSystem();
	const hkArray<hkpAction*>& action = system->getActions();

	for(int i=0; i < action.getSize(); i++)
	{
		actionAddedCallback(action[i]);
	}

	system->removeReference();

	w->unmarkForWrite();
}

void hkpVehicleViewer::worldRemovedCallback( hkpWorld* w)
{
	w->markForWrite();
	w->removeActionListener(this);
	w->removeWorldPostSimulationListener(this);
	w->unmarkForWrite();
}


// HVK-3274
// This function ensures that hkpVehicleViewer will fail to compile if the name
// of hkpVehicleInstance is ever changed. Should this ever happen the string
// below needs to be changed accordingly. This removes the dependency on
// hkvehicle from hkutilities.
static inline const char* getHkVehicleInstanceName(hkpVehicleInstance*) { return "hkpVehicleInstance"; }

void hkpVehicleViewer::actionAddedCallback( hkpAction* action )
{
	hkBuiltinTypeRegistry& registry = hkBuiltinTypeRegistry::getInstance();

	// Get the class of the action that was added to the world
	// NB: To get the class we use a string here, rather than
	// hkpVehicleInstanceClass, so that building hctUtilities does
	// not have a dependency on the vehicle library. See HVK-3274.
	const hkClass* vehicleClass = registry.getClassNameRegistry()->getClassByName(getHkVehicleInstanceName(HK_NULL));
	const hkClass* actionClass	= registry.getVtableClassRegistry()->getClassFromVirtualInstance(action);


	// If the action is a vehicle instance, or it
	// is of a type derived from the vehicle class.
	if (vehicleClass && actionClass && vehicleClass->isSuperClass(*actionClass))
	{
		action->addReference();
		m_vehicles.pushBack(static_cast<hkpVehicleInstance*>(action));
	}
}

void hkpVehicleViewer::actionRemovedCallback( hkpAction* action )
{
	// Search for the action that has just been removed from the
	// world. If the action is in our list of vehicles, remove it
	// from the list.
	const int index = m_vehicles.indexOf((hkpVehicleInstance*)action);

	// If this action is not a member of m_vehicles, we ignore it and return.
	if (index == -1) return;

	m_vehicles.removeAt(index);
	action->removeReference();
}

void hkpVehicleViewer::postSimulationCallback(hkpWorld* world)
{
	HK_TIME_CODE_BLOCK("hkpVehicleViewer::step", HK_NULL);

	const int numVehicles = m_vehicles.getSize();

	static const hkColor::Argb inAirColor		= hkColor::ROSYBROWN;
	static const hkColor::Argb inContactColor	= hkColor::WHITESMOKE;
	static const hkColor::Argb raycastColor	= hkColor::RED;
	static const hkColor::Argb axleColor		= hkColor::YELLOWGREEN;

	// Loop through each vehicle in the world
	for (int i=0; i<numVehicles; i++)
	{
		const hkpVehicleInstance& vehicle = *m_vehicles[i];

		// Loop through each wheel on the vehicle
		const int numWheels = vehicle.m_wheelsInfo.getSize();
		hkLocalBuffer<hkVector4> centerPoint(numWheels);
		for ( int w = 0; w < numWheels; w++ )
		{

			const hkpVehicleData::WheelComponentParams& wheelParam = vehicle.m_data->m_wheelParams[w];
			hkpVehicleSuspension::SuspensionWheelParameters& suspensionParam = vehicle.m_suspension->m_wheelParams[w];
			const hkpVehicleInstance::WheelInfo& wheel	= vehicle.m_wheelsInfo[w];
			const hkVector4& contactPoint				= wheel.m_contactPoint.getPosition();
			const hkVector4& hardPoint					= wheel.m_hardPointWs;

			const int collidableId = wheel.m_contactBody ? (int)(hkUlong)(wheel.m_contactBody->getCollidable()) : 0;

			centerPoint[w].setAddMul(hardPoint, wheel.m_suspensionDirectionWs, hkSimdReal::fromFloat(suspensionParam.m_length));

			// Draw a ray from hardPoint to center to contactPoint
			m_displayHandler->displayLine(hardPoint, centerPoint[w], wheel.m_contactBody ? raycastColor : inAirColor, collidableId, m_tag);
			m_displayHandler->displayLine(centerPoint[w], contactPoint, wheel.m_contactBody ? raycastColor : inAirColor, collidableId, m_tag);

			// Draw the wheel rotation line
			{
				hkVector4 spoke = wheel.m_suspensionDirectionWs;
				hkQuaternion spokeRotation; spokeRotation.setAxisAngle(wheel.m_spinAxisWs, -wheel.m_spinAngle);
				spoke.setRotatedDir(spokeRotation, spoke);
				spoke.normalize<3>();
				spoke.mul(hkSimdReal::fromFloat(wheelParam.m_radius));

				hkVector4 start;	start.setAdd(centerPoint[w], spoke);
				hkVector4 end;		end.setSub(centerPoint[w], spoke);

				m_displayHandler->displayLine(start, end, wheel.m_contactBody ? inContactColor : inAirColor, collidableId, m_tag);
			}

			// Draw the wheel
			hkVector4 spokes[s_numWheelSegments];
			spokes[0].setMul(hkSimdReal::fromFloat(wheelParam.m_radius), wheel.m_suspensionDirectionWs);

			// This loop defines a number of points on the circumference of the
			// wheel rim. It then draws a lines to connect all these points.
			hkQuaternion spokeRotation; spokeRotation.setAxisAngle(wheel.m_spinAxisWs, DEGREES2RADIANS(360.0f/(float)s_numWheelSegments));
			for (int s=1; s<s_numWheelSegments+2; s++)
			{
				if (s < s_numWheelSegments)
				{
					spokes[s].setRotatedDir(spokeRotation, spokes[s-1]);
				}
				if (s <= s_numWheelSegments)
				{
					spokes[s-1].add(centerPoint[w]);
				}
				if (s>=2)
				{
					m_displayHandler->displayLine(	spokes[(s-2)%s_numWheelSegments],
						spokes[(s-1)%s_numWheelSegments],
						wheel.m_contactBody ? inContactColor : inAirColor, collidableId, m_tag);
				}
			}
		}

		// Draw the axles (Do this by drawing a line between all wheels on the
		// same axle).
		const int numAxles  = vehicle.m_data->m_numWheelsPerAxle.getSize();
		for (int axle=0; axle < numAxles; axle++)
		{
			// These two vectors will be used to store the point of connected with
			// the axle of the last two wheels that we have encountered.
			hkVector4 a;
			hkVector4 b;

			// This flag will be used so that we know when we are going through
			// this loop for the first time.
			bool firstTimeThrough = true;
			const int totalWheelsOnThisAxle = vehicle.m_data->m_numWheelsPerAxle[axle];
			int wheelsFoundOnThisAxle  = 0;

			for ( int w = 0; w < numWheels; w++ )
			{
				const hkpVehicleData::WheelComponentParams& wheelParam = vehicle.m_data->m_wheelParams[w];
				const hkpVehicleInstance::WheelInfo& wheel	= vehicle.m_wheelsInfo[w];
				const int collidableId = wheel.m_contactBody ? (int)(hkUlong)(wheel.m_contactBody->getCollidable()) : 0;

				// If this wheel is not connected to the axle that we are
				// concerned with, skip it.
				if (wheelParam.m_axle != axle) continue;

				wheelsFoundOnThisAxle++;

				a = b;
				b = centerPoint[w];

				if (firstTimeThrough) 
				{
					// On this first time through this loop, the vector 'a'
					// would be uninitialized, so we do nothing.
					firstTimeThrough = false;
				}
				else
				{
					// Draw a line to connect the last 2 wheels that we
					// have encountered.
					m_displayHandler->displayLine(a, b, axleColor, collidableId, m_tag);
				}

				// If we have already dealt with all the wheels on this axle,
				// we can break from the loop.
				if (wheelsFoundOnThisAxle == totalWheelsOnThisAxle) break;
			}
		}	
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
