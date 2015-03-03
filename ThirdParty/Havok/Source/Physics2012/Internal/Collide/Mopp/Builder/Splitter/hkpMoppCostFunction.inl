/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


// costPrimitiveSplits
// Returns 0.0 if no primitives are split
// The two splitting planes can have arbitrary positions.
// If the two splitting planes both split a primitive, it is actually split and the cost is increased.
hkReal hkpMoppCostFunction::costPrimitiveSplits(const hkpMoppCostFunction::hkpPlanesParams& p)
{
	const hkReal splitVMCost = 5.0f;
	const hkReal cost = p.m_numSplitPrimitives * p.m_numPrimitivesInv;  
	return cost * splitVMCost * m_userScaling.m_weightPrimitiveSplit;
}

// cost for particular direction
hkReal hkpMoppCostFunction::directionCost(const hkpMoppCostFunction::hkpPlaneRightParams& planeRightInfo)
{
	return planeRightInfo.m_plane->m_cost;
}


// costPlaneDistance
// calculates the cost associated with the relative positions of the two splitting planes to each other
hkReal hkpMoppCostFunction::costPlaneDistance(const hkpMoppCostFunction::hkpPlanesParams& p)
{
	const hkReal distance = p.m_planeLeftPosition - p.m_planeRightPosition;
	const hkReal relCost = distance * p.m_extentsInv;

	return relCost * m_userScaling.m_weightPlaneDistance;
}


// costNumUnBalanced
// Returns 0.0 if there are the same amount of primitives on each side
hkReal hkpMoppCostFunction::costNumUnBalanced(const hkpMoppCostFunction::hkpPlanesParams& p)
{
	const int diffPrimitives = p.m_currentNumPrimitivesRight - p.m_currentNumPrimitivesLeft;
	const hkReal diff = hkMath::fabs( hkReal (diffPrimitives) );
	const hkReal costPre = diff * p.m_numPrimitivesInv;
	const hkReal maxZeroDistribution = 7.0f;
	const hkReal maxDiff = (maxZeroDistribution - 1.0f) / (maxZeroDistribution + 1.0f );

	hkReal cost;
	if ( p.m_currentDepth < p.m_optimalDepth)
	{
		// set cost to zero if the distribution is maxZeroDistribution:1
		cost = costPre - maxDiff;	
		cost *= 1.5f / (1.0f - maxDiff);
		if ( cost < 0.0f )
		{
			return 0.0f;
		}
	}
	else
	{
		cost = costPre * costPre;
	}
	const hkReal cost4 =  cost * cost * cost * cost;
	return cost4 * m_userScaling.m_weightNumUnbalancedTriangles;
}


// costUnbalancedPlaneRight ( for the left subtree )
// retruns 0.0 for optimal position
// bases cost on a hyperbolic function of the form f(x) = (1/x) - 2,
// such that f(0.0f ... 0.5f) = 0.0f
// and       f(0.5f ....1.0f) > 0.0f
hkReal hkpMoppCostFunction::costUnbalancedPlaneRight(const hkpMoppCostFunction::hkpPlaneRightParams& p)
{
	// scale x value using x->(pos-min)/(max-min) for [min,max]->[0,1]
	const hkReal center = 0.5f * ( p.m_absoluteMax + p.m_absoluteMin );
	const hkReal planePosition = p.m_planeRightPosition - center;

	// only allow for unbalanced trees if we are not too deep
	hkReal invExtents;
	if ( p.m_currentDepth < p.m_optimalDepth)
	{
		const hkReal extents = p.m_absoluteMax - p.m_absoluteMin;
		hkReal extra = hkMath::min2( m_userScaling.m_queryErrorTolerance, extents );
		if ( (extents + extra) > 0 )
		{
			invExtents = 1.0f / (extents + extra);
		}
		else
		{
			invExtents = HK_REAL_MAX * 0.1f;
		}
	}
	else
	{
		invExtents = p.m_extentsInv;
	}

	// set x to be in the range -0.5f ... 0.5f
	const hkReal x = planePosition * invExtents;

	if ( x < 0.0f)
	{
		return 0.0f;
	}
	const hkReal x2 = x * x;
	const hkReal x4 = x2 * x2;
	const hkReal x8 = x4 * x4;
	const hkReal cost = x2 * 2.9f + x4 * 3.0f + x8 * 1500.0f;

	return cost * m_userScaling.m_weightPlanePosition;
}

// costUnbalancedPlaneLeft (for the right subtree)
// returns 0.0 for optimal position
// bases cost on a hyperbolic function of the form f(x) = 1/(1-x) - 2,
//
// such that f(0.0f ... 0.5f) > 0.0f
// and       f(0.5f ....1.0f) = 0.0f

hkReal hkpMoppCostFunction::costUnbalancedPlaneLeft(const hkpMoppCostFunction::hkpPlanesParams& p)
{
	const hkReal center = 0.5f * ( p.m_absoluteMax + p.m_absoluteMin );
	const hkReal planePosition = p.m_planeLeftPosition - center;

	// only allow for unbalanced trees if we are not too deep
	hkReal invExtents;
	if ( p.m_currentDepth < p.m_optimalDepth)
	{
		const hkReal extents = p.m_absoluteMax - p.m_absoluteMin;
		hkReal extra = hkMath::min2( m_userScaling.m_queryErrorTolerance, extents );
		if ( (extents + extra) > 0 )
		{
			invExtents = 1.0f / (extents + extra);
		}
		else
		{
			invExtents = HK_REAL_MAX * 0.1f;
		}

	}
	else
	{
		invExtents = p.m_extentsInv;
	}

	// set x to be in the range -0.5f ... 0.5f
	const hkReal x = planePosition * invExtents;

	if ( x > 0.0f)
	{
		return 0.0f;
	}
	const hkReal x2 = x * x;
	const hkReal x4 = x2 * x2;
	const hkReal x8 = x4 * x4;
	const hkReal cost = x2 * 2.9f + x4 * 3.0f + x8 * 1500.0f;

	return cost * m_userScaling.m_weightPlanePosition;
}

hkReal hkpMoppCostFunction::costprimitiveIdSpread( const hkpMoppCostFunction::hkpPlaneRightParams& p )
{
	//if ( p.m_numPrimitives > 180 )
	//	return 0.0f;

	int spread = p.m_maxPrimitiveId - p.m_minPrimitiveId;
	if ( spread < 16 )
	{
		return -0.03f * m_userScaling.m_weightPrimitiveIdSpread;
	}
	return 0.0f;

	/*
	if ( spread < 256 )
	{
	return spread * m_userScaling.m_primitiveIdSpread * ( 0.1f / 256.0f);
	}

	return m_userScaling.m_primitiveIdSpread * 0.1f;
	*/
}



// only take cost of plane 1 position into account
hkReal hkpMoppCostFunction::costPlaneRight(const hkpMoppCostFunction::hkpPlaneRightParams& planeRightInfo)
{
	hkpMoppBasicNode::hkpMoppCostInfo ci;
	ci.m_planeRightPositionCost = costUnbalancedPlaneRight(planeRightInfo);	
	ci.m_directionCost = directionCost(planeRightInfo);
	ci.m_primitiveIdSpread = costprimitiveIdSpread(planeRightInfo);
	return ci.m_planeRightPositionCost + ci.m_directionCost + ci.m_primitiveIdSpread;
}

// total main cost function
hkReal hkpMoppCostFunction::extraCostPlaneLeft(const hkpMoppCostFunction::hkpPlanesParams& p2)
{
	hkpMoppBasicNode::hkpMoppCostInfo ci;
	ci.m_planeLeftPositionCost = costUnbalancedPlaneLeft(p2);
	ci.m_numUnBalancedCost  = costNumUnBalanced(p2);
	ci.m_splitCost          = costPrimitiveSplits(p2);
	ci.m_planeDistanceCost  = costPlaneDistance(p2);

	return ci.m_planeLeftPositionCost + ci.m_numUnBalancedCost + ci.m_splitCost + ci.m_planeDistanceCost;
}

#ifdef MOPP_DEBUG_COSTS
void hkpMoppCostFunction::debugCosts ( const hkpMoppCostFunction::hkpPlanesParams& p, hkpMoppBasicNode::hkpMoppCostInfo& ci )
{
	ci.m_planeRightPositionCost = costUnbalancedPlaneRight(p);	
	ci.m_planeLeftPositionCost = costUnbalancedPlaneLeft(p);
	ci.m_numUnBalancedCost  = costNumUnBalanced(p);
	ci.m_splitCost          = costPrimitiveSplits(p);
	ci.m_planeDistanceCost  = costPlaneDistance(p);
	ci.m_absoluteMin = p.m_absoluteMin;
	ci.m_absoluteMax = p.m_absoluteMax;
}
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
