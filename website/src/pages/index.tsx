import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

// --- Hero Section Component ---
function HomepageHeader() {
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <h1 className={styles.heroTitle}>
          <span>Pie:</span> Programmable LLM Serving
        </h1>
        <p className={styles.heroSubtitle}>
          Go beyond serving prompts.<br /> Define, optimize, and deploy your custom LLM inference workflows with Pie.
        </p>
        <div className={styles.heroLinks}>
          <Link className="button button--primary button--lg" to="https://github.com/pie-project/pie">
            View on GitHub
          </Link>
          <Link className="button button--secondary button--lg" to="/blog/announcing-pie">
            Announcement Post
          </Link>
        </div>
      </div>
    </header>
  );
}

// --- Main Home Page Component ---
export default function Home(): JSX.Element {
  return (
    <Layout
      title="Home"
      description="Pie is a programmable serving system for emerging LLM applications"
    >
      <HomepageHeader />
      <main>
        {/* Architecture Section */}
        <section className={clsx(styles.section, styles.architectureSection)}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Modern AI Demands A New Serving Paradigm</h2>
            <p className={styles.sectionSubtitle}>
              Current LLM serving systems use a rigid, monolithic loop,<br />creating bottlenecks for complex applications. Pie replaces this with a fully programmable architecture using sandboxed WebAssembly programs called <strong>inferlets</strong>.
            </p>

            <div className={styles.architectureGrid}>
              <div className={styles.archCard}>
                <h3>Existing LLM Serving</h3>
                <img
                  src={useBaseUrl('/img/current-serving.svg')}
                  alt="Diagram of a monolithic LLM serving architecture"
                />
                <p>
                  A one-size-fits-all process that limits innovation and forces inefficient workarounds for advanced use cases.
                </p>
              </div>

              <div className={styles.archCard}>
                <h3>Programmable LLM Serving</h3>
                <img
                  src={useBaseUrl('/img/programmable-serving.svg')}
                  alt="Diagram of Pie’s programmable serving architecture"
                />
                <p>
                  A flexible foundation of fine-grained APIs, giving you direct control to build application-specific optimizations.
                </p>
              </div>
            </div>
          </div>
        </section>


        {/* Contrast Section: Weakness → Pie Strength */}
        <section className={clsx(styles.section, styles.contrastSection)}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Serve Programs, Not Prompts</h2>
            <p className={styles.sectionSubtitle}>
              By moving control from the system to the developer-written <strong>inferlets</strong>,<br /> Pie unlocks new capabilities and optimizations for advanced LLM workflows.
            </p>

            <div className={styles.contrastGrid} role="list">
              {/* Row 1 */}
              <div className={styles.contrastRow}>
                {/* Weakness */}
                <article className={styles.weakCard} role="listitem" aria-labelledby="w1">
                  <p className={styles.cardEyebrow}>Existing Systems</p>
                  <h3 id="w1" className={styles.cardTitle}>
                    Inference inefficiency
                  </h3>
                  <p className={styles.cardBody}>
                    Missing application-level optimizations lead to wasted tokens, redundant compute, and rigid execution paths.
                  </p>
                  <div className={styles.mobileArrow} aria-hidden="true">↓ becomes ↓</div>
                </article>

                {/* Arrow */}
                <div className={styles.flowArrow} aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M5 12h14"></path>
                    <path d="m13 5 7 7-7 7"></path>
                  </svg>
                </div>

                {/* Strength */}
                <article className={styles.goodCard} aria-labelledby="s1">
                  <img
                    src={useBaseUrl('/img/pie-dark.svg')}
                    alt="Pie"
                    className={styles.cardEyebrowLogo}
                  />
                  <h3 id="s1" className={styles.cardTitle}>
                    🎛️ Fine-grained KV cache control
                  </h3>
                  <p className={styles.cardBody}>
                    Customize KV cache management around your reasoning pattern, prune or reuse states precisely, and avoid
                    unnecessary recompute.
                  </p>
                </article>
              </div>

              {/* Row 2 */}
              <div className={styles.contrastRow}>
                <article className={styles.weakCard} role="listitem" aria-labelledby="w2">
                  <p className={styles.cardEyebrow}>Existing Systems</p>
                  <h3 id="w2" className={styles.cardTitle}>
                    Implementation challenges
                  </h3>
                  <p className={styles.cardBody}>
                    Custom decoding and optimization methods require invasive system patches or forks, complicating maintenance and velocity.
                  </p>
                  <div className={styles.mobileArrow} aria-hidden="true">↓ becomes ↓</div>
                </article>

                <div className={styles.flowArrow} aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M5 12h14"></path>
                    <path d="m13 5 7 7-7 7"></path>
                  </svg>
                </div>

                <article className={styles.goodCard} aria-labelledby="s2">
                  <img
                    src={useBaseUrl('/img/pie-dark.svg')}
                    alt="Pie"
                    className={styles.cardEyebrowLogo}
                  />
                  <h3 id="s2" className={styles.cardTitle}>
                    ⚙️ Customizable generation
                  </h3>
                  <p className={styles.cardBody}>
                    Define bespoke decoding algorithms and safety filters per request using programmable APIs. No system fork required.
                  </p>
                </article>
              </div>

              {/* Row 3 */}
              <div className={styles.contrastRow}>
                <article className={styles.weakCard} role="listitem" aria-labelledby="w3">
                  <p className={styles.cardEyebrow}>Existing Systems</p>
                  <h3 id="w3" className={styles.cardTitle}>
                    Integration friction
                  </h3>
                  <p className={styles.cardBody}>
                    External data/tools sit outside the generation loop, adding round-trip latency and brittle glue code.
                  </p>
                  <div className={styles.mobileArrow} aria-hidden="true">↓ becomes ↓</div>
                </article>

                <div className={styles.flowArrow} aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M5 12h14"></path>
                    <path d="m13 5 7 7-7 7"></path>
                  </svg>
                </div>

                <article className={styles.goodCard} aria-labelledby="s3">
                  <img
                    src={useBaseUrl('/img/pie-dark.svg')}
                    alt="Pie"
                    className={styles.cardEyebrowLogo}
                  />
                  <h3 id="s3" className={styles.cardTitle}>
                    🔗 Seamless workflow integration
                  </h3>
                  <p className={styles.cardBody}>
                    Call tools and data sources inside the serving engine to cut round-trips and keep state aligned with execution.
                  </p>
                </article>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className={clsx(styles.section, styles.ctaSection)}>
          <div className="container text--center">
            <h2 className={styles.sectionTitle}>Dive Deeper</h2>
            <p className={styles.sectionSubtitle}>
              Ready to see how it works? Check out our documentation and get started with Pie today.
            </p>
            <div className={styles.ctaActions}>
              <Link className="button button--primary button--lg" to="/docs/installation">
                Getting Started
              </Link>

            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}