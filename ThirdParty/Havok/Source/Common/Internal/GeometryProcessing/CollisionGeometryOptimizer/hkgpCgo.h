/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKGP_CGO_H
#define HKGP_CGO_H

#include <Common/Base/Config/hkProductFeatures.h>

/// This is the hi-level interface to the collision geometry optimizer.
/// Note: This is NOT meant as a mesh simplifier.
/// WARNING: This feature is currently in BETA. Changes in interface, behaviour or performance may be expected.
struct hkgpCgo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpCgo);

	/// Tracker.
	struct Tracker
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,Tracker);

		virtual ~Tracker() {}
		virtual void	edgeFlip(int vertexFrom, int vertexTo) = 0;
		virtual void	edgeCollapse(int vertexFrom, int vertexTo, const hkVector4& position, hkReal error) = 0;
		virtual void	setClusterId(const int* triangleIndices, int numTriangles, int cid) {}
	};

	/// Configuration.
	struct Config
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,Config);

		/// Defaults
		Config()
		{			
			m_semantic					=	VS_NONE;
			m_combinator				=	VC_MIN;
			m_maxDistance				=	HK_REAL_MAX;
			m_maxShrink					=	hkReal(0.0f);
			m_maxAngle					=	HK_REAL_MAX;
			m_minEdgeRatio				=	hkReal(0.0f);
			m_maxAngleDrift				=	hkReal(HK_REAL_PI);
			m_weldDistance				=	hkReal(0.0f);
			m_updateThreshold			=	hkReal(0.0f);
			m_maxVertices				=	0;
			m_inverseOrientation		=	false;
			m_proportionalShrinking		=	false;
			m_multiPass					=	false;
			m_multiThreaded				=	true;
			m_protectNakedBoundaries	=	false;
			m_protectMaterialBoundaries	=	false;
			m_decimateComponents		=	false;
			m_tracker					=	HK_NULL;
			
			m_solverAccuracy			=	SA_NORMAL;
			m_minDistance				=	hkReal(-0.0001f);
			m_minConvergence			=	hkReal(0.00001f);
			m_project					=	true;
			m_buildClusters				=	true;
			m_useAccumulatedError		=	false;
			m_useLegacySolver			=	false;
		}

		/// Semantic of the W component of vertices.
		enum VertexSemantic
		{
			VS_NONE,		///< No semantic.
			VS_WEIGHT,		///< Max distance fraction as in 'Config::m_minDistance + (Config::m_maxDistance - Config::m_minDistance) * W'.
			VS_DISTANCE,	///< Max distance.
			VS_FACTOR,		///< Error * W.
			VS_OFFSET		///< Error + W.
		};

		/// Combinator for the W components of vertices.
		enum VertexCombinator
		{
			VC_MIN,			///< Min of W0 and W1.
			VC_MAX,			///< Max of W0 and W1.
			VC_MEAN			///< (W0 + W1) / 2.
		};

		/// Solver accuracy setting.
		enum SolverAccuracy
		{
			SA_FAST		=	4,
			SA_NORMAL	=	8,
			SA_ACCURATE	=	16,
			SA_HIGH		=	64,
		};
		
		// Optimization parameters.

		VertexSemantic		m_semantic;					///< Vertex semantic.
		VertexCombinator	m_combinator;				///< Vertex combinator.
		hkReal				m_maxDistance;				///< Maximum allowed distance to the mesh surface (unit: distance).
		hkReal				m_maxShrink;				///< Maximum allowed shrinking (unit: fraction between 0 and 1).
		hkReal				m_maxAngle;					///< Protect concave edges that form an angle greater than this value (unit: radian betwen 0 and pi).
		hkReal				m_minEdgeRatio;				///< Minimum allowed triangle edge length ratio (min edge length divided by max edge length, default 0).
		hkReal				m_maxAngleDrift;			///< Maximum allowed angle drift (unit: radian betwen 0 and pi).
		hkReal				m_weldDistance;				///< Automaticaly weld vertices below or equal to this distance (unit: distance), set to a negative number to disable.
		hkReal				m_updateThreshold;			///< Perform multiple optimization up to this threshold (delta between current and previous error).
		int					m_maxVertices;				///< Maximum allowed number of vertices.
		hkBool				m_inverseOrientation;		///< Invert the mesh orientation, thus shrink instead of expanding.
		hkBool				m_proportionalShrinking;	///< Shrinking is proportional to error.
		hkBool				m_multiPass;				///< Allow repeated optimization until no further simplification can be applied.
		hkBool				m_multiThreaded;			///< Use multiple threads.
		hkBool				m_protectNakedBoundaries;	///< Do not optimize open boundaries if true.
		hkBool				m_protectMaterialBoundaries;///< Do not optimize across material boundaries if true.
		hkBool				m_decimateComponents;		///< Decimate components smaller the current error.
		Tracker*			m_tracker;					///< Tracker.

		// Internals.
		SolverAccuracy	m_solverAccuracy;
		hkReal			m_minDistance;
		hkReal			m_minConvergence;
		hkBool			m_project;		
		hkBool			m_buildClusters;
		hkBool			m_useAccumulatedError;
		hkBool			m_useLegacySolver;
	};

	/// Progress interface.
	struct IProgress
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,IProgress);

		virtual			~IProgress()	{}

		/// Called after before optimization step, return if the process should continue (true) or not (false).
		virtual bool	step(const Config& config, hkReal error, int numVertices, int numTriangles)=0;
	};

	/// ClusterData, use for analysis only.
	struct ClusterData
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,ClusterData);

		HK_FORCE_INLINE	bool	operator<(const ClusterData& other) const { return m_errorRange[0] < other.m_errorRange[0]; }

		hkReal			m_errorRange[2];
		int				m_numVertices;
	};

	/// Optimize geometry.
	/// Note: If a non-null array of clusters if passed, optimize will only perform a analysis and fill clusters->getSize() clusters data but do not changes the geometry.
	static void HK_CALL optimize(const Config& config, struct hkGeometry& geometry, IProgress* progress = HK_NULL, hkArray<ClusterData>* clusters = HK_NULL);

	// Utility class
protected:
	hkgpCgo()	{}
	hkgpCgo(const hkgpCgo&)	{}
};

#endif // HKGP_CGO_H

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
