/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/TyreMarks/hknpTyremarksInfo.h>


hknpTyremarksWheel::hknpTyremarksWheel()
{
	m_numPoints = 0;
	m_currentPosition = 0;
	m_tyremarkPoints.clear();
}

void hknpTyremarksWheel::setNumPoints( int num_points )
{
	HK_ASSERT(0x6203f15d, num_points > 0 );
	m_numPoints = num_points;
	m_currentPosition = 0;

	for ( int i = 0 ; i < m_numPoints ; i++ )
	{
		hknpTyremarkPoint tyrePoint;
		m_tyremarkPoints.pushBack( tyrePoint );
	}
}

hknpTyremarkPoint::hknpTyremarkPoint()
{
	m_pointLeft.setZero();
	m_pointRight.setZero();
}

const hknpTyremarkPoint& hknpTyremarksWheel::getTyremarkPoint( int point) const
{
	const int index = ( m_currentPosition + point) % m_numPoints;
	// we add m_numPoints for the case m_currentPosition == point == 0
	// also due to the fact that we are using unsigned ints
	return m_tyremarkPoints[index];
}

hkReal hknpTyremarkPoint::getTyremarkStrength() const
{
	 return m_pointRight(3);
}

void hknpTyremarksWheel::addTyremarkPoint( hknpTyremarkPoint& point)
{
	const int previous_index = (m_currentPosition + m_numPoints - 1) % m_numPoints;

	const hkSimdReal previous_strength = m_tyremarkPoints [previous_index].m_pointRight.getW();

	if (point.m_pointRight.getW().isEqualZero() && previous_strength.isEqualZero())
	{
		return; // two tyremarks == 0.0f in a row, we can ignore this one;
	}

	m_tyremarkPoints[m_currentPosition] = point;

	m_currentPosition = ( m_currentPosition + 1 ) % m_numPoints;
}

hknpTyremarksInfo::hknpTyremarksInfo(const hknpVehicleData& data, int num_points)
{
	m_minTyremarkEnergy = 0;
	m_maxTyremarkEnergy = 0;

	m_tyremarksWheel.setSize( data.m_numWheels );
	for (int i = 0; i < m_tyremarksWheel.getSize() ; i++ )
	{
		hknpTyremarksWheel* tyremarksWheel = new hknpTyremarksWheel();
		tyremarksWheel->setNumPoints(num_points);
		m_tyremarksWheel[i] = tyremarksWheel;
	}
}

hknpTyremarksInfo::~hknpTyremarksInfo()
{
	for ( int sw_it = 0 ; sw_it < m_tyremarksWheel.getSize() ; sw_it++ )
	{
		m_tyremarksWheel[sw_it]->removeReference();
	}
}


void hknpTyremarksInfo::updateTyremarksInfo( hkReal timestep, const hknpVehicleInstance* vehicle )
{
	const hkTransform &car_transform = vehicle->getChassisTransform();

	hkVector4 offset;
	offset.setMul( hkSimdReal::fromFloat(timestep), vehicle->getChassisMotion().getLinearVelocity() );


	const hkVector4 &right_cs = vehicle->m_data->m_chassisOrientation.getColumn<2>();
	hkVector4 right_ws;
	right_ws.setRotatedDir(car_transform.getRotation(),right_cs);

	int w_it;
	for (w_it=0; w_it< vehicle->m_data->m_numWheels; w_it++)
	{
		const hknpVehicleInstance::WheelInfo &wheel_info = vehicle->m_wheelsInfo[w_it];

		hkSimdReal tyre_alpha = hkSimdReal::fromFloat(wheel_info.m_skidEnergyDensity);
		const hkSimdReal minE = hkSimdReal::fromFloat(m_minTyremarkEnergy);
		const hkSimdReal maxE = hkSimdReal::fromFloat(m_maxTyremarkEnergy);
		tyre_alpha.sub(minE);
		if (tyre_alpha.isGreaterZero())
		{
			tyre_alpha = tyre_alpha * hkSimdReal_255 / ( maxE - minE ); // scaled between 0.0f and 255.0f

			tyre_alpha.setMax(tyre_alpha, hkSimdReal_255);
		}
		else
		{
			tyre_alpha.setZero();
		}

		const hkSimdReal wheel_width = hkSimdReal::fromFloat(vehicle->m_data->m_wheelParams[w_it].m_width);
		const hkVector4 &contact_point_ws = wheel_info.m_contactPoint.getPosition();
		const hkVector4 &normal_ws = wheel_info.m_contactPoint.getNormal();

		hkVector4 a_little_up;
		a_little_up.setMul(hkSimdReal::fromFloat(0.05f), normal_ws);

		hkVector4 point_ws;			point_ws.setAdd( contact_point_ws, a_little_up);
		hkVector4 left;				left.setMul( -hkSimdReal_Inv2 * wheel_width, right_ws);
		hkVector4 point_left_ws; 	point_left_ws.setAdd ( point_ws, left);
		hkVector4 right;			right.setMul( hkSimdReal_Inv2 * wheel_width, right_ws);
		hkVector4 point_right_ws;	point_right_ws.setAdd( point_ws, right);

		point_left_ws.add(offset);

		hknpTyremarkPoint new_point;

#if defined(_MSC_VER) && (_MSC_VER >= 1200) && (_MSC_VER < 1300)
		// work around an ICE in vc6
		point_right_ws(0) += offset(0);
		point_right_ws(3) += offset(3);

		{
			new_point.m_pointLeft    = point_left_ws;
			new_point.m_pointLeft(3) = tyre_alpha;
			new_point.m_pointRight   = point_right_ws;
			new_point.m_pointRight(1) = point_right_ws(1) + offset(1);
			new_point.m_pointRight(2) = point_right_ws(2) + offset(2);

			new_point.m_pointRight(3) = tyre_alpha;
		}
#else
		point_right_ws.add(offset);

		{
			new_point.m_pointLeft.setXYZ_W(point_left_ws, tyre_alpha);
			new_point.m_pointRight.setXYZ_W(point_right_ws, tyre_alpha);
		}
#endif

		m_tyremarksWheel[w_it]->addTyremarkPoint(new_point);
	}
}


void hknpTyremarksInfo::getWheelTyremarksStrips( const hknpVehicleInstance* vehicle, int wheel, hkVector4 *strips_out) const
{

	const hknpTyremarksWheel &tyremarks_wheel = *m_tyremarksWheel[wheel];
	const int num_points = tyremarks_wheel.m_numPoints;

	for (int p_it=0; p_it< num_points; p_it++)
	{
		const hknpTyremarkPoint &tyre_point = tyremarks_wheel.getTyremarkPoint(p_it);

		strips_out [p_it * 2] = tyre_point.m_pointLeft;
		strips_out [p_it * 2 +1] = tyre_point.m_pointRight;
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
