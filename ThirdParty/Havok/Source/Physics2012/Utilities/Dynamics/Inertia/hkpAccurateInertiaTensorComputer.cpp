/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpAccurateInertiaTensorComputer.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

#include <Common/Base/Config/hkOptionalComponent.h>

HK_OPTIONAL_COMPONENT_DEFINE(hkpAccurateInertiaTensorComputer, hkpInertiaTensorComputer::s_computeConvexHullMassPropertiesFunction, hkpAccurateInertiaTensorComputer::computeConvexHullMassProperties);

hkResult HK_CALL hkpAccurateInertiaTensorComputer::computeConvexHullMassProperties(const hkStridedVertices& vertices, hkReal r, hkMassProperties& result)
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpAccurateInertiaTensorComputer);

	/* Build the convex hull of vertices	*/ 
	hkgpConvexHull hull;
	hkgpConvexHull::BuildConfig	config;
	config.m_allowLowerDimensions			=	true;
	config.m_alwaysComputeProjectionPlane	=	false;
	hull.build(vertices,config);
	
  
	const int dim = hull.getDimensions();
	const hkReal minAllowedRadius = hkReal(100) * HK_REAL_EPSILON;
	const hkReal minAllowedVolume = hkReal(100) * HK_REAL_EPSILON;
	hkSimdReal skinVolume; skinVolume.setZero();

	const hkSimdReal radius = hkSimdReal::fromFloat(r);
	const hkReal hullRadius = hkMath::max2(r,minAllowedRadius);

    switch ( dim )
    {
        case 0:
        {
            // We have only one vertex and a radius : it's a sphere !
            return hkpInertiaTensorComputer::computeSphereVolumeMassProperties( hullRadius, hkReal(1), result );
        }

        case 1:
        {
            // We have a straight line and a radius : it's a capsule !
            hkVector4 a, b;
			a.load<3,HK_IO_NATIVE_ALIGNED>(vertices.m_vertices);
			b.load<3,HK_IO_NATIVE_ALIGNED>(hkAddByteOffsetConst(vertices.m_vertices,vertices.m_striding));
			const int numVertices = vertices.getSize();
			
			// If there are more than 2 vertices (aligned in a straight line), 
			// we need to find the outermost vectors to make our capsule axis.
			if (numVertices > 2 )
			{
				hkVector4 dir, current;
				dir.setSub(b,a); // axis direction

				// Find the minimum and maximum dot product values.
				int minIdx = 0;
				int maxIdx = 0 ;
				hkSimdReal minValue; minValue.setZero();
				hkSimdReal maxValue; maxValue.setZero();

				for (int i = 1; i < numVertices; ++i) 
				{
					current.load<3,HK_IO_NATIVE_ALIGNED>(hkAddByteOffsetConst(vertices.m_vertices,i*vertices.m_striding));
					const hkSimdReal distance = dir.dot<3>(current);
					
					if (distance > maxValue)
					{
                        maxIdx = i; 
                        maxValue = distance;
					}
					
					if (distance < minValue)
					{
                        minIdx = i;
                        minValue = distance; 
					}
				}

				HK_ASSERT2(0x16ba3e40, minIdx != maxIdx, "Did not find capsule axis !");
				a.load<3,HK_IO_NATIVE_ALIGNED>(hkAddByteOffsetConst(vertices.m_vertices,minIdx*vertices.m_striding));
				b.load<3,HK_IO_NATIVE_ALIGNED>(hkAddByteOffsetConst(vertices.m_vertices,maxIdx*vertices.m_striding));
            }

            return hkpInertiaTensorComputer::computeCapsuleVolumeMassProperties( a, b, hullRadius, hkReal(1), result );
        }

		case 2:
		{
			// We grow the (plane) shape by a small margin, trying  
			// to take the radius into account if there is one.
            hkReal margin = hkMath::max2(hkConvexShapeDefaultRadius*hkReal(0.2f), radius.getReal());

            /* Expand by 'margin' until we can compute mass properties	*/ 
            do 
            {
                hull.expandByPlanarMargin(margin);
                if(hull.getDimensions() == 3)
				{ 
					hull.buildMassProperties();
                }
				margin *= hkReal(2);
			} while( hull.getDimensions() != 3 || hull.getVolume() < hkSimdReal::fromFloat(minAllowedVolume) );
			
			break;
		}

		case 3:
		{  
			hull.buildMassProperties();
			/* Expand by 'radius'									*/ 
			if(radius.isGreaterZero())
			{
				/* Add prims volume									*/ 
				skinVolume	=	hull.getSurfaceArea()*radius;
				/* Add sphere volume								*/ 
				skinVolume.add(hkSimdReal::fromFloat(hkReal(4)/hkReal(3)*HK_REAL_PI)*radius*radius*radius);
			}
			break;
		}

		default :
		{
			return HK_FAILURE;
		}
    
    } // end of switch statement

    /* Set the resulting mass properties						*/ 
    result.m_centerOfMass	=	hull.getCenterOfMass();
    result.m_inertiaTensor	=	hull.getWorldInertia();
	const hkSimdReal resultVol = hull.getVolume()+skinVolume;
	resultVol.store<1>(&result.m_volume);
    result.m_mass			=	hkReal(1);
    if(resultVol.isGreaterZero() && skinVolume.isGreaterZero())
    {
		hkSimdReal scale; scale.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(skinVolume + skinVolume, hkSimdReal_3 * resultVol);
        result.m_inertiaTensor.mul(hkSimdReal_1 + scale);
    }
	return HK_SUCCESS;
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
