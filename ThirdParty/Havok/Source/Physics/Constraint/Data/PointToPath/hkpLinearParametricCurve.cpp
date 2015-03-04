/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics/Constraint/Data/PointToPath/hkpLinearParametricCurve.h>

hkpLinearParametricCurve::hkpLinearParametricCurve()
:	m_smoothingFactor( 0.01f ), m_closedLoop( false )
{
	m_dirNotParallelToTangentAlongWholePath = hkVector4::getConstant<HK_QUADREAL_0100>();
}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::getPoint( hkReal t, hkVector4& pointOnCurve ) const
{
	int ia = hkMath::max2( 0, hkMath::hkFloatToInt(t) );
	int ib = ia + 1;
	hkSimdReal ia_f; ia_f.setFromInt32(ia);
	hkSimdReal t_f; t_f.setFromFloat(t);

	if( ib < m_points.getSize() )
	{
		pointOnCurve.setInterpolate(m_points[ia], m_points[ib], t_f - ia_f);
	}
	else
	{
		pointOnCurve = m_points[ia];
	}
}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::setClosedLoop( hkBool closeLoop )
{
	m_closedLoop = closeLoop;
}

//////////////////////////////////////////////////////////////////////

hkBool hkpLinearParametricCurve::isClosedLoop() const
{
	return m_closedLoop;
}


//////////////////////////////////////////////////////////////////////

hkReal hkpLinearParametricCurve::getNearestPoint( hkReal t, const hkVector4& nearPoint, hkVector4& pointOnCurve ) const
{

	HK_ASSERT(0x5967b43a,  m_points.getSize() > 1 );

	int ia = hkMath::max2( 0, hkMath::hkFloatToInt(t) );
	int ib = ia + 1;

	if( ib >= m_points.getSize() )
	{
		ib = m_points.getSize()-1;
		ia = ib - 1;
	}

	hkSimdReal nearProjAB; nearProjAB.setZero();

	// search for straddling points
	while( 1 )
	{

		hkVector4 tangentAB;	tangentAB.setSub( m_points[ib], m_points[ia]);

		//
		//	Check for first side
		//

		hkVector4 fromA;	fromA.setSub( nearPoint, m_points[ia]);

		nearProjAB = tangentAB.dot<3>( fromA );
		// this will be a # between 0 and 1 if the point lies between them, it can be < 0 or > 1
		// at boundary cases
		const hkSimdReal tanLenSqrd = tangentAB.dot<3>(tangentAB);
		nearProjAB.div(tanLenSqrd);

		if( nearProjAB.isLessZero() )
		{
			if ( ia == 0)
			{
				break;
			}
			else
			{
				ia--;
				ib--;
				continue;
			}
		}

		hkVector4 fromB;		fromB.setSub( nearPoint, m_points[ib]);
		const hkSimdReal checkEndAB = tangentAB.dot<3>( fromB );

		if ( checkEndAB.isLessEqualZero() )
		{
			break;
		}

		int ic = ib + 1;
		if( ic >= m_points.getSize() )
		{
			break;
		}

		hkVector4 tangentBC;	tangentBC.setSub( m_points[ic], m_points[ib]);

		const hkSimdReal nearProjBC = tangentBC.dot<3>( fromB );
		if( nearProjBC.isGreaterZero() )
		{
			ia++;
			ib++;
			continue;
		}

		{  // ack, it's in the reflex elbo of a seam

			// find the segment it is angularly closer to and make the point lay there
			tangentAB.normalize<3>();
			tangentBC.normalize<3>();

			const hkSimdReal angleToAB = fromB.dot<3>( tangentAB );
			const hkSimdReal angleToBC = fromB.dot<3>( tangentBC );

			if( -angleToBC > angleToAB )
			{
				nearProjAB.setFromFloat(0.99f);
				break;
			}
			else
			{
				ia++;
				ib++;
				tangentAB = tangentBC;
				nearProjAB.setFromFloat(0.01f);
				break;
			}
		}
	}


	pointOnCurve.setInterpolate( m_points[ia],  m_points[ib], nearProjAB);

	hkReal new_parametric_value = hkReal(ia) + nearProjAB.getReal();

	// for a closed loop the last segment should overlap the first segment exactly
	// we place two transition points where the body jumps from one end of the path to the other
	if( m_closedLoop )
	{
		hkReal endPoint = hkReal( m_points.getSize()-1 );
		if( new_parametric_value < hkReal(0.25f) )
		{
			new_parametric_value = endPoint - ( hkReal(1) - new_parametric_value );
			new_parametric_value = getNearestPoint( new_parametric_value, nearPoint, pointOnCurve );
		}
		else if( new_parametric_value > endPoint - hkReal(0.25f) )
		{
			new_parametric_value = hkReal(1) - ( endPoint - new_parametric_value );
			new_parametric_value = getNearestPoint( new_parametric_value, nearPoint, pointOnCurve );
		}
	}

	return new_parametric_value;

}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::getTangent( hkReal t, hkVector4& tangent ) const
{

	hkSimdReal smooth_tolerance; smooth_tolerance.setFromFloat(m_smoothingFactor);
	hkSimdReal smooth_tolerance_inv; smooth_tolerance_inv.setReciprocal(smooth_tolerance);
	hkSimdReal tt; tt.setFromFloat(t);

	int t_i;
	tt.storeSaturateInt32(&t_i);

	int ia = hkMath::max2( 0, t_i );
	int ib = ia + 1;

	if( ib >= m_points.getSize() )
	{
		ib = m_points.getSize() - 1;
		ia = ib - 1;
	}

	tangent.setSub( m_points[ib], m_points[ia]);
	tangent.normalize<3>();

	hkSimdReal t_remainder = tt - hkSimdReal::fromInt32(ia);
	// if we are near a seam, smooth it out a bit
	if( t_remainder < smooth_tolerance && ia > 0 )
	{
		hkVector4 tangent2;

		ia--;
		ib--;

		tangent2.setSub( m_points[ib], m_points[ia]);
		tangent2.normalize<3>();

		const hkSimdReal interp = hkSimdReal::getConstant<HK_QUADREAL_INV_2>() * ( smooth_tolerance - t_remainder ) * smooth_tolerance_inv;
		tangent.setInterpolate( tangent,  tangent2, interp );
		tangent.normalize<3>();
	}

	t_remainder = hkSimdReal::fromInt32(ib) - tt;
	// if we are near a seam, smooth it out a bit
	if( t_remainder < smooth_tolerance && ib < m_points.getSize()-1 )
	{
		hkVector4 tangent2;

		ia++;
		ib++;

		tangent2.setSub( m_points[ib], m_points[ia]);
		tangent2.normalize<3>();

		const hkSimdReal interp = hkSimdReal::getConstant<HK_QUADREAL_INV_2>() * ( smooth_tolerance - t_remainder ) * smooth_tolerance_inv;
		tangent.setInterpolate( tangent, tangent2, interp );
		tangent.normalize<3>();
	}

}

//////////////////////////////////////////////////////////////////////

hkReal hkpLinearParametricCurve::getStart() const
{
	return hkReal(0);
}

//////////////////////////////////////////////////////////////////////

hkReal hkpLinearParametricCurve::getEnd() const
{
	return hkReal( m_points.getSize()-1 );
}

//////////////////////////////////////////////////////////////////////

hkReal hkpLinearParametricCurve::getLengthFromStart( hkReal t ) const
{
	int ia = hkMath::max2( 0, hkMath::hkFloatToInt(t) );
	hkReal segment_scale = hkReal(0);

	if( ia >= m_points.getSize()-1 )
	{
		ia = m_points.getSize()-1;

		segment_scale = m_distance[ia] - m_distance[ia-1];

	}
	else
	{
		segment_scale = m_distance[ia+1] - m_distance[ia];
	}

	hkReal dist = m_distance[ia] + (t-hkReal(ia))*segment_scale;

	return dist;
}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::getBinormal( hkReal t, hkVector4& up ) const
{
	hkVector4 tangent;
	getTangent( t, tangent );

	if( tangent.dot<3>( hkTransform::getIdentity().getColumn<1>() ).getReal() < 0.98f  )
	{
		up.setCross(tangent,m_dirNotParallelToTangentAlongWholePath);
	}
	else
	{
		hkVector4Util::calculatePerpendicularVector( tangent, up );
	}

	up.normalize<3>();
}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::addPoint(const hkVector4& p )
{

	m_points.pushBack( p );

	if( m_points.getSize() == 1 )
	{
		m_distance.pushBack( hkReal(0) );
	}
	else
	{
		hkVector4 p0 = m_points[ m_points.getSize() - 2 ];
		hkVector4 delta;
		delta.setSub( p, p0);

		hkReal last_dist = m_distance[ m_distance.getSize() - 1 ];
		m_distance.pushBack( delta.length<3>().getReal() + last_dist );
	}

}

//////////////////////////////////////////////////////////////////////

hkReal hkpLinearParametricCurve::getSmoothingFactor() const
{
	return m_smoothingFactor;
}


//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::setSmoothingFactor( hkReal smooth )
{
	m_smoothingFactor = smooth;
}

//////////////////////////////////////////////////////////////////////

void hkpLinearParametricCurve::getPointsToDraw(hkArray<hkVector4>& pathPoints) const
{
	// Okay so we are doing a copy. Could pass pointer, but
	// that interface assumes that all path types maintain a list of points
	// used for display.
	pathPoints = m_points;
}



////////////////////////////////////////////////////////////////////
///Transform all the points in the curve
void hkpLinearParametricCurve::transformPoints( const hkTransform& transformation)
{
	const int numPoints = m_points.getSize();

	for( int i = 0; i < numPoints; i++)
	{
		m_points[i]._setTransformedPos(transformation,m_points[i]);
	}
}


//////////////////////////////////////////////////////
///Create an exact copy of the curve
hkpParametricCurve* hkpLinearParametricCurve::clone()
{
	hkpLinearParametricCurve* newCurve = new hkpLinearParametricCurve();

	const int size =	  m_points.getSize();
	for (int i = 0; i < size; i++)
	{
		newCurve->addPoint(m_points[i]);
	}

	newCurve->m_closedLoop = m_closedLoop;
	newCurve->m_dirNotParallelToTangentAlongWholePath.setXYZ_0(m_dirNotParallelToTangentAlongWholePath);
	newCurve->m_distance = m_distance;
	newCurve->m_smoothingFactor = m_smoothingFactor;

	return newCurve;
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
